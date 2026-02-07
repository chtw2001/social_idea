import numpy as np
# import bottleneck as bn
import torch
import math
import pandas as pd
import os
from numba import njit, prange


def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    # GroundTruth(list of list)를 Numba가 좋아하는 CSR 형태(1D array + offsets)로 변환
    gt_indices = []
    gt_indptr = [0]
    for items in GroundTruth:
        gt_indices.extend(items)
        gt_indptr.append(len(gt_indices))
    
    gt_indices = np.array(gt_indices, dtype=np.int32)
    gt_indptr = np.array(gt_indptr, dtype=np.int32)
    predictedIndices = np.array(predictedIndices, dtype=np.int32)
    topN = np.array(topN, dtype=np.int32)
    
    # 실제 계산 (Numba 함수 호출)
    # 병렬 처리는 복잡하므로 여기서는 안전하게 단일 스레드 고속 연산(nopython=True)만 사용해도 충분히 빠름
    p, r, n, m, h = _compute_numba_inner(predictedIndices, gt_indices, gt_indptr, topN)
    
    return p.tolist(), r.tolist(), n.tolist(), m.tolist(), h.tolist()

@njit(nopython=True)
def _compute_numba_inner(preds, gt_indices, gt_indptr, top_k_arr):
    n_users = len(preds)
    n_k = len(top_k_arr)
    
    # Accumulators
    sum_prec = np.zeros(n_k, dtype=np.float64)
    sum_recall = np.zeros(n_k, dtype=np.float64)
    sum_ndcg = np.zeros(n_k, dtype=np.float64)
    sum_mrr = np.zeros(n_k, dtype=np.float64)
    sum_hr = np.zeros(n_k, dtype=np.float64)
    
    for i in range(n_users):
        start = gt_indptr[i]
        end = gt_indptr[i+1]
        if start == end: continue # No ground truth
        
        gt_items = gt_indices[start:end]
        n_gt = end - start
        
        for k_i in range(n_k):
            k = top_k_arr[k_i]
            # Top-K slicing
            curr_preds = preds[i, :k]
            
            hits = 0
            dcg = 0.0
            idcg = 0.0
            mrr = 0.0
            
            # Hit & DCG & MRR
            for r in range(k):
                p_item = curr_preds[r]
                # Linear scan for hit check (usually faster than set for small n_gt)
                is_hit = False
                for g_item in gt_items:
                    if p_item == g_item:
                        is_hit = True
                        break
                
                if is_hit:
                    hits += 1
                    dcg += 1.0 / np.log2(r + 2.0)
                    if mrr == 0.0:
                        mrr = 1.0 / (r + 1.0)
            
            # IDCG
            limit = min(n_gt, k)
            for r in range(limit):
                idcg += 1.0 / np.log2(r + 2.0)
                
            sum_prec[k_i]   += hits / k
            sum_recall[k_i] += hits / n_gt
            sum_ndcg[k_i]   += (dcg / idcg) if idcg > 0 else 0.0
            sum_mrr[k_i]    += mrr
            sum_hr[k_i]     += 1.0 if hits > 0 else 0.0
            
    return sum_prec/n_users, sum_recall/n_users, sum_ndcg/n_users, sum_mrr/n_users, sum_hr/n_users



def print_results(loss, valid_result, test_result):
    """output the evaluation results."""
    if loss is not None:
        print("[Train]: loss: {:.4f}".format(loss))
    if valid_result is not None: 
        print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {} HR:{}".format(
                            '-'.join([str(x) for x in valid_result[0]]), 
                            '-'.join([str(x) for x in valid_result[1]]), 
                            '-'.join([str(x) for x in valid_result[2]]), 
                            '-'.join([str(x) for x in valid_result[3]]),
                            '-'.join([str(x) for x in valid_result[4]])))
    if test_result is not None: 
        print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {} HR:{}".format(
                            '-'.join([str(x) for x in test_result[0]]), 
                            '-'.join([str(x) for x in test_result[1]]), 
                            '-'.join([str(x) for x in test_result[2]]), 
                            '-'.join([str(x) for x in test_result[3]]),
                            '-'.join([str(x) for x in test_result[4]])))
        
def metric_to_df(test_result,Ks):
    metric_names = ['ndcg','recall','precision','mrr','hr']
    metric_col=['K']+metric_names
    ndcg=[];recall=[];precision=[];mrr=[];hr=[]
    k=[]
    for i,k_value in enumerate(Ks):
        k.append([k_value])
        recall.append(test_result[1][i])
        ndcg.append(test_result[2][i])
        precision.append(test_result[0][i])
        mrr.append(test_result[3][i])
        hr.append(test_result[4][i])
    metrics_df=pd.DataFrame([k,ndcg,recall,precision,mrr,hr]).transpose()
    metrics_df.columns=metric_col
    return metrics_df

def save_results(test_result,dir,Ks,name=None):

    metric_df=metric_to_df(test_result,Ks)
    if name is None:
        metric_df.to_csv(dir + '/test_int_metrics.tsv', sep='\t', index=False)
    else:
        metric_df.to_csv(dir + name, sep='\t', index=False)

def save_model(model, model_dir, current_epoch,last_best_epoch=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(current_epoch))
    torch.save({'model_state_dict': model.state_dict(), 'epoch': current_epoch}, model_state_file)

    if last_best_epoch is not None and current_epoch != last_best_epoch:
        old_model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(last_best_epoch))
        if os.path.exists(old_model_state_file):
           # print("exist and remove"+old_model_state_file)
            os.system('rm {}'.format(old_model_state_file))

def evaluate(model, args, data, test=False, precomputed=None, train=False):
    u_batch_size = data.test_batch_size
    
    # [수정 1] train=True일 때 train_user_dict를 사용하도록 분기 처리
    if train:
        testDict = data.train_user_dict  # Data Loader에 이 변수가 있어야 합니다.
        mode = "train"
    elif test:
        testDict = data.test_user_dict
        mode = "test"
    else:
        testDict = data.valid_user_dict
        mode = "valid"

    # [수정 2] 캐시 키에 mode('train', 'test', 'valid')를 사용하여 구분
    cache = getattr(data, "_eval_cache", None)
    if cache is None:
        data._eval_cache = {}
        cache = data._eval_cache
        
    cache_key = (mode, u_batch_size) # mode 변수 사용
    
    if cache_key in cache:
        user_ids_batches, target_items = cache[cache_key]
    else:
        user_ids = list(testDict.keys())
        user_ids_batches = [user_ids[i: i + u_batch_size] for i in range(0, len(user_ids), u_batch_size)]
        user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]
        target_items = [ testDict[user_ids[i]] for i in range(len(user_ids))]
        cache[cache_key] = (user_ids_batches, target_items)
        
    topN = eval(args.topN)
    model.eval()

    # [핵심] precomputed가 있으면 그대로 사용 (Train도 동일하게 적용됨)
    if precomputed is None:
        with torch.no_grad():
            all_users, all_items = model.infer_embedding()
    else:
        all_users, all_items = precomputed
        
    all_items_t = all_items.t()

    predict_items = []
    with torch.no_grad():
        for batch in user_ids_batches:
            allPos = data.getUserPosItems(batch)
            batch = batch.to(all_users.device)
            user_embed = all_users[batch]
            prediction = torch.mm(user_embed, all_items_t)
            if not train: # Train 평가가 아닐 때만 이미 본 아이템 마스킹
                exclude_index = []
                exclude_items = []
                for range_i, items in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                prediction[exclude_index, exclude_items] = -(1<<10)
            
            _, indices = torch.topk(prediction, topN[-1])
            indices = indices.cpu().numpy().tolist()
            predict_items.extend(indices) 

    test_results = computeTopNAccuracy(target_items, predict_items, topN)

    return test_results