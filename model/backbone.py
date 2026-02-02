import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import coo_matrix
from utils.set_seed import set_seed, set_numba_seed

def to_tensor(coo_mat, device):
    """Scipy Sparse Matrix를 Torch Sparse Tensor로 변환"""
    values = coo_mat.data
    indices = np.vstack((coo_mat.row, coo_mat.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo_mat.shape
    tensor_sparse = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return tensor_sparse.to(device)

class SocialLightGCN(nn.Module):
    def __init__(self, data, args):
        super(SocialLightGCN, self).__init__()
        set_seed(args.seed)
        set_numba_seed(args.seed)
        self.config = args
        self.n_layers = self.config.n_layers
        self.emb_size = self.config.embed_dim
        self.data = data
        self.device = args.device

        self.num_users = self.data.n_users
        self.num_items = self.data.n_items

        # 임베딩 초기화 (LightGCN 스타일: 정규분포 0.1)
        self.user_embeddings = nn.Embedding(self.num_users, self.emb_size)
        self.item_embeddings = nn.Embedding(self.num_items, self.emb_size)
        nn.init.normal_(self.user_embeddings.weight, std=0.1)
        nn.init.normal_(self.item_embeddings.weight, std=0.1)

        # 그래프 초기화
        self.init_graph()

    def init_graph(self):
        """초기 그래프 생성"""
        # 1. Social Graph (H_s): main.py의 DSM이 업데이트하는 대상
        self.H_s = self.buildSparseRelationMatrix()
        self.H_s = to_tensor(self.H_s, self.device)

        # 2. Interaction Graph (R): User-Item 상호작용
        # LightGCN은 보통 (N+M)x(N+M) 큰 인접행렬을 쓰지만, 
        # MHCN 구조 호환성을 위해 R(User->Item)과 R_t(Item->User) 분리 방식을 유지합니다.
        self.R_mat = self.buildJointAdjacency() # Scipy Matrix
        self.R = to_tensor(self.R_mat, self.device) # User -> Item Aggregation용
        self.R_t = to_tensor(self.R_mat.transpose(), self.device) # Item -> User Aggregation용

    def buildSparseRelationMatrix(self):
        """소셜 그래프 생성 (main.py의 로직과 호환)"""
        row = np.array(self.data.train_social_h_list)
        col = np.array(self.data.train_social_t_list)
        
        # 가중치 처리
        if hasattr(self.data, 'train_social_w_list') and self.data.train_social_w_list is not None:
            min_len = min(len(row), len(self.data.train_social_w_list))
            row = row[:min_len]
            col = col[:min_len]
            # LightGCN 백본에서는 가중치를 그대로 사용 (Binary화 하지 않음)
            entries = np.array(self.data.train_social_w_list[:min_len], dtype=np.float32)
        else:
            entries = np.ones(len(row), dtype=np.float32)

        # Row-Normalization (GCN 기본)
        adj = coo_matrix((entries, (row, col)), shape=(self.num_users, self.num_users), dtype=np.float32)
        
        # Self-loop 추가 여부는 선택적이나, 보통 소셜 전파 시 본인 정보 유지를 위해 추가하기도 함.
        # 여기서는 MHCN.py와의 일관성을 위해 Row-Normalize만 수행
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = coo_matrix((d_inv, (np.arange(self.num_users), np.arange(self.num_users))), shape=(self.num_users, self.num_users))
        norm_adj = d_mat.dot(adj).tocoo()
        return norm_adj

    def buildJointAdjacency(self):
        """User-Item Interaction Graph 생성 (Normalized)"""
        # LightGCN의 D^-0.5 A D^-0.5 정규화 방식 적용
        row = np.array(self.data.train_h_list)
        col = np.array(self.data.train_t_list)
        entries = np.ones(len(row), dtype=np.float32)

        # User-Item Adjacency
        adj = coo_matrix((entries, (row, col)), shape=(self.num_users, self.num_items), dtype=np.float32)
        
        # Degree 계산
        user_degree = np.array(adj.sum(axis=1)).flatten()
        item_degree = np.array(adj.sum(axis=0)).flatten()
        
        # 0으로 나누기 방지
        user_degree[user_degree == 0] = 1e-12
        item_degree[item_degree == 0] = 1e-12

        d_u_inv = np.power(user_degree, -0.5)
        d_i_inv = np.power(item_degree, -0.5)
        
        # D_u^-0.5 * R * D_i^-0.5
        # row(user)에 d_u_inv 곱하기
        entries = entries * d_u_inv[row] * d_i_inv[col]
        
        norm_adj = coo_matrix((entries, (row, col)), shape=(self.num_users, self.num_items))
        return norm_adj

    def update_channels_from_S_fast(self, rebuild_R=False):
        """
        main.py에서 DSM 업데이트 후 호출하는 함수.
        소셜 그래프(H_s)를 갱신합니다.
        """
        # 소셜 그래프 재구성
        self.H_s = self.buildSparseRelationMatrix()
        self.H_s = to_tensor(self.H_s, self.device)
        
        if rebuild_R:
            self.R_mat = self.buildJointAdjacency()
            self.R = to_tensor(self.R_mat, self.device)
            self.R_t = to_tensor(self.R_mat.transpose(), self.device)

    def infer_embedding(self):
        """LightGCN 스타일 전파 + 소셜 전파"""
        users_emb = self.user_embeddings.weight
        items_emb = self.item_embeddings.weight
        
        all_users = [users_emb]
        all_items = [items_emb]

        for layer in range(self.n_layers):
            # 1. User-Item Interaction Propagation (LightGCN)
            # User <-> Item 양방향 전파
            # New User = R * Item
            # New Item = R^T * User
            new_users_rec = torch.sparse.mm(self.R, items_emb)
            new_items_rec = torch.sparse.mm(self.R_t, users_emb)
            
            # 2. Social Propagation (User-User)
            # New User Social = H_s * User
            new_users_soc = torch.sparse.mm(self.H_s, users_emb)
            
            # 3. Aggregation (Fusion)
            # Interaction 정보와 Social 정보를 합침.
            # 단순 합(Sum) 혹은 평균 사용. 여기서는 Social 영향력을 1.0으로 가정하고 더함.
            # (필요시 args.social_weight 같은 파라미터로 조절 가능)
            users_emb = (1 - self.config.social_weight)*new_users_rec + self.config.social_weight*new_users_soc 
            items_emb = new_items_rec
            
            all_users.append(users_emb)
            all_items.append(items_emb)

        # Layer Averaging (LightGCN 최종 단계)
        final_users = torch.stack(all_users, dim=1).mean(dim=1)
        final_items = torch.stack(all_items, dim=1).mean(dim=1)

        return final_users, final_items

    def bpr_loss(self, u_idx, v_idx, neg_idx):
        """
        BPR Loss 계산
        Return: (final_loss, reg_loss, ss_loss) -> ss_loss는 0 반환 (단순화)
        """
        final_user_embeddings, final_item_embeddings = self.infer_embedding()

        # Embedding Lookup
        user_emb = final_user_embeddings[u_idx]
        pos_emb = final_item_embeddings[v_idx]
        neg_emb = final_item_embeddings[neg_idx]

        # Score Calculation (Inner Product)
        pos_scores = torch.sum(user_emb * pos_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=1)

        # BPR Loss: -log(sigmoid(pos - neg)) = softplus(neg - pos)
        loss = torch.mean(F.softplus(neg_scores - pos_scores))

        # Regularization (L2)
        # 초기 임베딩(self.user_embeddings)에 대해 규제하는 것이 일반적
        user_emb_0 = self.user_embeddings(u_idx)
        pos_emb_0 = self.item_embeddings(v_idx)
        neg_emb_0 = self.item_embeddings(neg_idx)
        
        reg_loss = (1/2) * (user_emb_0.norm(2).pow(2) + 
                            pos_emb_0.norm(2).pow(2) + 
                            neg_emb_0.norm(2).pow(2)) / float(len(u_idx))

        # Total Loss
        # main.py에서 loss + reg * lambda 형태로 계산하므로 여기서는 term만 반환
        # ss_loss는 사용하지 않으므로 0.0 반환
        return loss, reg_loss, 0.0

    def forward(self, user_id, item_id):
        """평가 시 사용"""
        all_user, all_item = self.infer_embedding()
        user_emb = all_user[user_id]
        item_emb = all_item[item_id]
        cf_scores = torch.mul(user_emb, item_emb).sum(dim=1)
        return cf_scores