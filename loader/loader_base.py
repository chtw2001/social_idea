from cgi import test
import os
from ssl import PROTOCOL_TLS_CLIENT
import scipy.sparse as sp
import numpy as np
import pandas as pd
import torch.utils.data as data
from utils.set_seed import set_seed, set_numba_seed

class DataLoaderBase(data.Dataset):

    def __init__(self, args, logging):
        self.args = args
        set_seed(self.args.seed)
        set_numba_seed(args.seed)
        self.data_name = args.dataset
        self.device=args.device
        self.data_dir = os.path.join(args.data_path, args.dataset)
        self.num_ng=1

        self.train_file = os.path.join(self.data_dir, 'train_list.npy')
        self.valid_file = os.path.join(self.data_dir, 'valid_list.npy')
        self.test_file = os.path.join(self.data_dir, 'test_list.npy')
        self.social_file = os.path.join(self.data_dir, 'social_list.npy')
        
        self.cf_train_data, self.train_user_dict = self.load_data(self.train_file)
        self.cf_valid_data, self.valid_user_dict = self.load_data(self.valid_file)
        self.cf_test_data, self.test_user_dict = self.load_data(self.test_file)
        self.social_data, self.social_dict = self.load_data(self.social_file)

        self.statistic_cf()


    def ng_sample(self): 
        self.train_fill=[]
        for x in range(len(self.cf_train_data[0])):
            u, i = self.cf_train_data[0][x], self.cf_train_data[1][x]
            for t in range(self.num_ng):
                j = np.random.randint(self.n_items)
                while  j in self.train_user_dict[u]:
                    j = np.random.randint(self.n_items)
                self.train_fill.append([u, i, j])

    def __len__(self):     
   				 
        return self.num_ng * len(self.cf_train_data[0]) 
    
    def __getitem__(self,idx):
        user = self.cf_train_data[0][idx] # self.train_fill[idx][0]
        item_i = self.cf_train_data[1][idx] # self.train_fill[idx][1]
        item_j = 1 # self.train_fill[idx][2]
        return user, item_i, item_j 
    
    def load_data(self, filename):    

        train_list = np.load(filename, allow_pickle=True)

        user = train_list[:,0]
        item = train_list[:,1]
        user_dict = dict()

        for uid, iid in train_list:
            if uid not in user_dict:
                user_dict[uid] = []
            user_dict[uid].append(iid)
        return (user, item), user_dict
 
    def statistic_cf(self):
        a=[max(self.cf_train_data[0]), max(self.cf_test_data[0]),max(self.cf_valid_data[0]),
        max(self.social_data[0]),max(self.social_data[1])]
        b=[max(self.cf_train_data[1]), max(self.cf_test_data[1]),max(self.cf_valid_data[1])]
        self.n_users = max(a) + 1
        self.n_items = max(b) + 1
        self.n_cf_train = len(self.cf_train_data[0])
        self.n_cf_valid = len(self.cf_valid_data[0])
        self.n_cf_test = len(self.cf_test_data[0])
        
    def getUserPosItems(self, users):
        # -----------------------------------------------------------
        # [디버깅 코드 시작] 첫 번째 배치 호출 시에만 상위 100명 출력
        # -----------------------------------------------------------
        # if not hasattr(self, '_debug_print_done'):
        #     print("\n" + "="*50)
        #     print("[DEBUG] getUserPosItems: Checking first batch users...")
        #     print(f"Total Users in Batch: {len(users)}")
            
        #     count = 0
        #     for user in users:
        #         uid = int(user)
        #         is_in_train = uid in self.train_user_dict
                
        #         # 존재하는 유저는 'OK', 없으면 'MISSING' 출력
        #         status = "OK" if is_in_train else "🚨 MISSING (No Train Data)"
        #         item_count = len(self.train_user_dict.get(uid, []))
                
        #         print(f"User ID: {uid:<6} | Status: {status:<25} | Train Items: {item_count}")
                
        #         count += 1
        #         if count >= 100: # 100명까지만 보고 중단
        #             break
            
        #     print("="*50 + "\n")
        #     self._debug_print_done = True
        # -----------------------------------------------------------

        posItems = []
        for user in users:
            uid = int(user)
            # .get()을 사용하여 에러를 방지하되, 위 디버그 로그로 원인을 파악합니다.
            posItems.append(self.train_user_dict.get(uid, []))
            
        return posItems


import numpy as np
from numba import njit
import torch

# Numba로 컴파일된 고속 샘플링 코어 함수
@njit
def sample_negatives_numba(users, n_items, gt_indices, gt_indptr, n_neg):
    """
    users: 이번 배치의 유저 ID들 (int array)
    n_items: 전체 아이템 수
    gt_indices, gt_indptr: 유저별 Positive 아이템 목록 (CSR 형태)
    n_neg: 유저당 뽑을 네거티브 수
    """
    n_batch = len(users)
    neg_items = np.empty((n_batch, n_neg), dtype=np.int32)
    
    for i in range(n_batch):
        u = users[i]
        # 유저 u의 positive 아이템 범위
        start = gt_indptr[u]
        end = gt_indptr[u+1]
        
        for k in range(n_neg):
            while True:
                # 랜덤 아이템 생성
                neg_i = np.random.randint(0, n_items)
                
                # Positive인지 확인 (Linear Scan - Positive가 아주 많지 않으면 이게 더 빠름)
                is_pos = False
                for idx in range(start, end):
                    if gt_indices[idx] == neg_i:
                        is_pos = True
                        break
                
                if not is_pos:
                    neg_items[i, k] = neg_i
                    break
    return neg_items

class BatchSampler:
    def __init__(self, args, dataset, n_neg=1):
        self.n_items = dataset.n_items
        self.n_neg = n_neg
        set_seed(args.seed)
        set_numba_seed(args.seed)
        
        # 데이터셋의 train_user_dict를 CSR 형태의 Numpy Array로 변환 (Numba용)
        # SRDataLoader의 train_user_dict 사용
        user_dict = dataset.train_user_dict
        n_users = dataset.n_users
        
        indices = []
        indptr = [0] * (n_users + 1)
        
        cnt = 0
        for u in range(n_users):
            if u in user_dict:
                items = user_dict[u]
                indices.extend(items)
                cnt += len(items)
            indptr[u+1] = cnt
            
        self.gt_indices = np.array(indices, dtype=np.int32)
        self.gt_indptr = np.array(indptr, dtype=np.int32)

    def sample(self, users):
        """
        users: Tensor or List of user IDs
        Return: (n_batch, n_neg) negative items
        """
        if isinstance(users, torch.Tensor):
            users_np = users.cpu().numpy().astype(np.int32)
        else:
            users_np = np.array(users, dtype=np.int32)
            
        neg_items = sample_negatives_numba(
            users_np, self.n_items, self.gt_indices, self.gt_indptr, self.n_neg
        )
        return torch.from_numpy(neg_items)
