import argparse
from ast import parse
import os
import time
import numpy as np
import wandb
import scipy.sparse as sp

import torch  
import torch.optim as optim

from model.MHCN import *
from utils.evaluate_utils import *
from utils.log_helper import *
from loader.data_loader import SRDataLoader
from loader.loader_base import BatchSampler
from copy import deepcopy
import torch.utils.data as torch_data
import random
from utils.set_seed import set_seed, set_numba_seed
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# DSM 관련 import
from idea_DSM import DifferentiableSocialMask, spmat_to_edge_index, add_toppct_edges_from_embeddings



def _print_dsm_one_line(epoch, dsm, *, grad_norm=None, prev_z_mean=None, prev_z=None, eps=1e-4):
    with torch.no_grad():
        z = dsm.z.detach()
        S = torch.sigmoid(z / dsm.t)
        zm, zs = z.mean().item(), z.std().item()
        zmn, zmx = z.min().item(), z.max().item()
        
        Sm, Ss = S.mean().item(), S.std().item()
        Smn, Smx = S.min().item(), S.max().item()

        slo = (S < 0.01).float().mean().item() * 100.0
        shi = (S > 0.99).float().mean().item() * 100.0

        # 기본값
        dmu = float('nan'); dz_mae = float('nan'); dz_l2 = float('nan'); moved = float('nan')
        if prev_z is not None:
            delta = (z - prev_z)
            dz_mae = delta.abs().mean().item()
            dz_l2  = (delta.norm().item() / (delta.numel() ** 0.5))
            moved  = (delta.abs() > eps).float().mean().item() * 100.0
        if prev_z_mean is not None:
            dmu = zm - prev_z_mean

        gnm = grad_norm if (grad_norm is not None) else float('nan')
        E = z.numel()
        print(f"[DSM] ep={epoch} E={E} | z: μ={zm:.4f} σ={zs:.4f} min={zmn:.3f} max={zmx:.3f} "
              f"| S: μ={Sm:.4f} σ={Ss:.4f} min={Smn:.3f} max={Smx:.3f} sat_lo={slo:.1f}% sat_hi={shi:.1f}% \n"
              f"     | ‖∇z‖={gnm:.4e} | Δμ={dmu:.4f} | Δz_mae={dz_mae:.3e} Δz_l2={dz_l2:.3e} moved>{eps:.0e}:{moved:.1f}%")


def save_best_embeddings(model, save_dir, seed, precomputed=None):
    # 모델을 평가 모드로 전환 (Dropout 등 비활성화)
    model.eval()
    with torch.no_grad():
        # 임베딩 추출
        if precomputed is None:
            user_emb, item_emb = model.infer_embedding()
        else:
            user_emb, item_emb = precomputed
        
        # CPU로 이동 및 Numpy 변환
        user_emb_np = user_emb.cpu().numpy()
        item_emb_np = item_emb.cpu().numpy()
        
        # 저장 경로 설정 및 저장
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        np.save(os.path.join(save_dir, f'best_user_emb_{seed}.npy'), user_emb_np)
        np.save(os.path.join(save_dir, f'best_item_emb_{seed}.npy'), item_emb_np)
        print(f"Saved best embeddings to {save_dir}")


def save_best_social_graph(dsm, save_dir, seed):
    """최고 성능일 때 소셜 그래프 저장"""
    with torch.no_grad():
        # DSM에서 현재 소셜 그래프 추출
        ei, w = dsm.current_adjacency()
        edge_index_np = ei.detach().cpu().numpy()
        weights_np = w.detach().cpu().numpy()
        
        # 저장 경로 설정 및 저장
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # edge_index와 weights를 하나의 파일로 저장
        np.savez(
            os.path.join(save_dir, f'best_social_graph_{seed}.npz'),
            edge_index=edge_index_np,
            weights=weights_np
        )
        print(f"Saved best social graph to {save_dir}")


def train(args,log_path):
    
    if args.wandb:
        wandb.init(
            project="ides-MHCN",
            name=f"{args.dataset}_{args.idea_lr}_{args.sig_temp}_{args.threshold}_{args.weight_mode}",
            tags=["edge_add"]
        )
        
    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)
    device = args.device
    print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    data = SRDataLoader(args, logging)
    train_loader = torch_data.DataLoader(data,batch_size=args.batch_size, shuffle=True, pin_memory=True) # , worker_init_fn=worker_init_fn, generator=torch.Generator().manual_seed(args.seed))
    sampler = BatchSampler(args, data, n_neg=args.neg_num)
    print('data ready.')

    # DSM 초기화 (args.update_social == 1일 때만)
    dsm = None
    dsm_optimizer = None
    rebuild_channels_from_dsm = None
    
    if args.update_social == 1:
        print(">>> [DSM] 소셜 그래프 업데이트 활성화")
        # 소셜 그래프 가중치 초기화
        w_list = [1.0] * len(data.train_social_h_list)
        data.train_social_w_list = w_list
        social_edge_index = spmat_to_edge_index(data.social_graph).to(device)
        
        # DSM 생성
        dsm = DifferentiableSocialMask(
            args,
            num_nodes=data.n_users,
            observed_edge_index=social_edge_index,
            t=args.sig_temp,
            init_p1=0.9,
            device=device,
        )
        
        # DSM optimizer 생성
        dsm_optimizer = optim.Adam(dsm.parameters(), lr=args.idea_lr)
        prev_z = None  # DSM 모니터링용
        

    model = MHCN(data,args).to(device)
    
    if args.update_social == 1:
        
        # 채널 재구성 함수 정의 (model 생성 후)
        def _rebuild_channels_from_dsm():
            """DSM에서 업데이트된 소셜 그래프로 채널 재구성"""
            ei, w = dsm.current_adjacency()
            u = ei[0].detach().cpu().numpy()
            v = ei[1].detach().cpu().numpy()
            s = w.detach().cpu().numpy()
            
            # 작은 가중치 필터링
            mask = s > 1e-4
            if mask.sum() > 0:
                u, v, s = u[mask], v[mask], s[mask]
            
            data.train_social_h_list = u.tolist()
            data.train_social_t_list = v.tolist()
            data.train_social_w_list = s.tolist()
            model.update_channels_from_S_fast(rebuild_R=False)
        
        rebuild_channels_from_dsm = _rebuild_channels_from_dsm
        
    # threshold에 따라 pretrain edge 추가 (model 생성 후)
    if args.threshold > 0:
        # dsm이 없어도 엣지 추가 가능 (ablation study용)
        if dsm is None:
            # train_social_w_list가 없으면 초기화
            if not hasattr(data, 'train_social_w_list') or data.train_social_w_list is None:
                data.train_social_w_list = [1.0] * len(data.train_social_h_list)
        
        print(f">>> [Edge Addition] threshold={args.threshold}%로 pretrain edge 추가 시도")
        # 임베딩 파일 경로 확인 (여러 경로 시도)
        emb_path = f"./pretrained_emb/user_{args.dataset}.pkl"
        emb_found = False
        if os.path.exists(emb_path):
            print(f">>> [Edge Addition] 임베딩 파일 발견: {emb_path}")
            try:
                added_count = add_toppct_edges_from_embeddings(
                    dsm=dsm,
                    data=data,
                    device=device,
                    emb_path=emb_path,
                    top_pct=args.threshold,
                    weight_mode=args.weight_mode,
                )
                
                if args.wandb:
                    wandb.log({"pretrain_edges_added": added_count})
                
                emb_found = True
            except Exception as e:
                print(f">>> [Edge Addition] 오류 발생: {e}")
        
        if not emb_found:
            print(f">>> [Edge Addition] Warning: 임베딩 파일을 찾을 수 없습니다. 다음 경로를 확인했습니다:")
            print(">>> [Edge Addition] 엣지 추가를 건너뜁니다.")
        else:
            # 엣지 추가 후 모델 채널 업데이트
            if dsm is None:
                # dsm이 없을 때는 data를 직접 수정했으므로 모델 채널 업데이트
                model.update_channels_from_S_fast(rebuild_R=False)
                print(">>> [Edge Addition] 모델 채널 업데이트 완료")
            
    # 엣지 추가 후 optimizer 재초기화 (파라미터가 변경되었을 수 있음)
    if args.threshold > 0 and dsm is not None:
        with torch.no_grad():
            rebuild_channels_from_dsm()
        dsm_optimizer = optim.Adam(dsm.parameters(), lr=args.idea_lr)
    
    rec_optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print("models ready.")

    best_recall, best_epoch = -100, 0
    print("Start training...")
    set_seed(args.seed)
    set_numba_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        # train_loader.dataset.ng_sample()
        if epoch - best_epoch >= args.stopping_steps:
            print('-'*18)
            print('Exiting from training early')
            break

        print("Training SR")
        model.train()
        start_time = time.time()
        total_loss = 0.0
        epoch_start_z = None
            
        if args.update_social == 1:
            epoch_start_z = dsm.z.detach().clone()
            with torch.no_grad():
                rebuild_channels_from_dsm()
        
        for batch_user, batch_pos_item, _ in train_loader:
            batch_user = batch_user.long().to(device)
            batch_pos_item = batch_pos_item.long().to(device)
            # batch_neg_item = batch_neg_item.long().to(device)
            batch_neg_item = sampler.sample(batch_user).squeeze(1).long().to(device)
            
            # DSM 사용 시 H_s 업데이트
            if args.update_social == 1:
                model.H_s = dsm.to_sparse_coo(normalized="row")
            
            rec_optimizer.zero_grad()
            if args.update_social == 1:
                dsm_optimizer.zero_grad()
            
            losses = model.bpr_loss(batch_user, batch_pos_item, batch_neg_item)
            loss = losses[0]+losses[1]*args.lambda1+losses[2]*args.lambda2
            total_loss += loss.item()
            loss.backward()
            
            # DSM gradient norm 계산 (step 이전)
            dsm_grad_norm = None
            if args.update_social == 1 and dsm.z.grad is not None:
                dsm_grad_norm = dsm.z.grad.norm().item()
            
            
            rec_optimizer.step()
            if args.update_social == 1:
                dsm_optimizer.step()
        print('Backbone SR'+"Training Epoch {:03d} ".format(epoch) +'train loss {:.4f}'.format(total_loss) + " costs " + time.strftime(
                            "%H: %M: %S", time.gmtime(time.time()-start_time)))
        print('---'*18)
        
        # 평가 시에도 DSM 업데이트된 H_s 사용
        if args.update_social == 1:
            model.eval()
            with torch.no_grad():
                model.H_s = dsm.to_sparse_coo(normalized="row")

        model.eval()
        with torch.no_grad():
            all_users, all_items = model.infer_embedding()

        valid_results = evaluate(model,args,data,test=False, precomputed=(all_users, all_items))
        test_results = evaluate(model,args,data,test=True, precomputed=(all_users, all_items))
        print_results(None, valid_results, test_results)
        
        # DSM 모니터링 (5 epoch마다)
        if args.update_social == 1 and epoch % 5 == 0:
            _print_dsm_one_line(
            epoch, dsm,
            grad_norm=dsm_grad_norm if 'dsm_grad_norm' in locals() else None,
            prev_z_mean=epoch_start_z.mean().item() if epoch_start_z is not None else None,
            prev_z=epoch_start_z  # 시작 시점의 z를 넘김
        )
        
        if args.wandb:
            wandb.log({
                "val/epoch": epoch,
                "val/backboneloss": total_loss,
                "val/Precision@10": valid_results[0][0],
                "val/Recall@10": valid_results[1][0],
                "val/NDCG@10": valid_results[2][0],
                "val/MRR@10": valid_results[3][0],
                "val/HR@10": valid_results[4][0],
                "val/Precision@20": valid_results[0][1],
                "val/Recall@20": valid_results[1][1],
                "val/NDCG@20": valid_results[2][1],
                "val/MRR@20": valid_results[3][1],
                "val/HR@20": valid_results[4][1],
                "test/epoch": epoch,
                "test/Precision@10": test_results[0][0],
                "test/Recall@10": test_results[1][0],
                "test/NDCG@10": test_results[2][0],
                "test/MRR@10": test_results[3][0],
                "test/HR@10": test_results[4][0],
            })

        if valid_results[1][1] > best_recall: 
            save_model(model, args.save_dir, epoch,best_epoch)
            best_recall, best_epoch = valid_results[1][1], epoch
            best_results = valid_results
            best_test_results = test_results
            save_best_embeddings(model, args.save_dir, args.seed, precomputed=(all_users, all_items))
            if args.update_social == 1:
                save_best_social_graph(dsm, args.save_dir, args.seed)
            print('Save model on epoch {:04d}!'.format(epoch))
            
            if args.wandb:
                wandb.summary["best/epoch"] = best_epoch
                wandb.summary["best/Precision@10"] = best_test_results[0][0]
                wandb.summary["best/Recall@10"] = best_test_results[1][0]
                wandb.summary["best/NDCG@10"] = best_test_results[2][0]
                wandb.summary["best/MRR@10"] = best_test_results[3][0]
                wandb.summary["best/HR@10"] = best_test_results[4][0]

    #save diffusion model
    model_state_file = os.path.join(args.save_dir, 'diffusion.pth')
    torch.save({'model_state_dict': model.state_dict()}, model_state_file)
    print('==='*18)
    print("End. Best Epoch {:03d} ".format(best_epoch))
    print_results(None, best_results, best_test_results)   
    print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    ks=eval(args.topN)
    save_results(best_test_results,args.save_dir,ks)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='filmtrust')
    parser.add_argument('--data_path', type=str, default='datasets/')
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--stopping_steps', type=int, default=50)
    parser.add_argument('--topN', type=str, default='[10,20,50]')
    parser.add_argument('--device', nargs='?', default=1,type=int)  

    # parameter for sr backbone
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--lambda1', default=0.1,type=float)
    parser.add_argument('--lambda2', default=0.01,type=float)
    parser.add_argument('--neg_num', nargs='?', default=1) 

    # parameter for diffusion model
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--threshold', type=float, default=0.0, help='threshold for pretrain edge addition (percentage)')
    parser.add_argument('--weight_mode', type=str, default='cosine')
    parser.add_argument('--ss_every', type=int, default=1, help='compute self-supervised loss every N steps (1 = every step)')
    
    # DSM 관련 파라미터
    parser.add_argument('--update_social', type=int, default=0, help='1: Enable DSM-based social graph update')
    parser.add_argument('--sig_temp', type=float, default=0.001, help='sigmoid temperature for DSM')
    parser.add_argument('--idea_lr', type=float, default=0.001, help='learning rate for DSM optimizer')
    
    parser.add_argument('--wandb', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed) 
    set_numba_seed(args.seed)
    
    dataset =  args.dataset
    if dataset == 'filmtrust':
        args.n_layers = 2
        args.lr = 1e-2
        args.lambda1 = 1e-1
        args.lambda2 = 5e-2
    elif dataset == 'lastfm':
        args.n_layers = 3
        args.lr = 1e-4
        args.lambda1 = 1e-1
        args.lambda2 = 1e-2
    elif dataset == 'ciao':
        args.n_layers = 2
        args.lr = 1e-4
        args.lambda1 = 1e-2
        args.lambda2 = 5e-2
    elif dataset == 'douban':
        args.n_layers = 2
        args.lr = 1e-4
        args.lambda1 = 1e-1
        args.lambda2 = 1e-2
    elif dataset == 'yelp':
        args.n_layers = 2
        args.lr = 5e-5
        args.lambda1 = 1e-1
        args.lambda2 = 2.5e-1
    elif dataset == 'epinions':
        args.n_layers = 2
        args.lr = 1e-4
        args.lambda1 = 1e-1
        args.lambda2 = 1e-2
        
    
    
    args.device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")

    save_dir = 'saved_model/{}/idea-MHCN/idea_lr{}_sig_temp{}_thr_{}_{}/'.format(
    args.dataset,args.idea_lr,args.sig_temp, args.threshold, args.weight_mode)
    args.save_dir = save_dir

    log_path= 'saved_model/{}/idea-MHCN/idea_lr{}_sig_temp{}_thr_{}_{}/'.format(
    args.dataset,args.idea_lr,args.sig_temp, args.threshold, args.weight_mode)
    
    train(args,log_path)
