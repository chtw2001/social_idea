# idea_DSM.py
from typing import Optional, Tuple
import torch
from torch import nn
import numpy as np, scipy.sparse as sp
from utils.set_seed import set_seed, set_numba_seed

class DifferentiableSocialMask(nn.Module):
    """
    (한국어)
    - 관측된 소셜 엣지들에 대해 엣지별 확률(게이트) S_e ∈ (0,1)을 학습
    - S = sigmoid(z / t) (z: 학습 파라미터, t: 온도)
    - edge_index는 [2, E] 형태의 유향 엣지 목록
    """

    def __init__(
        self,
        args,
        num_nodes: int,
        observed_edge_index: torch.Tensor,
        *,
        t: float = 1.0,        
        init_p1: float = 0.9,  
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        assert observed_edge_index.dim() == 2 and observed_edge_index.size(0) == 2, "edge_index must be [2, E]"
        set_seed(args.seed)
        set_numba_seed(args.seed)
        self.num_nodes = int(num_nodes)
        self.t = float(t)

        edge_index = observed_edge_index.to(device=device)
        self.register_buffer("edge_index", edge_index)
        E = self.edge_index.size(1)

        # z 초기화: torch.logit 사용 (eps 설정으로 수치 안정성 확보)
        init_tensor = torch.tensor(init_p1, device=device, dtype=dtype)
        z_init_val = torch.logit(init_tensor, eps=1e-4).item() * self.t
        
        z = torch.full((E,), z_init_val, dtype=dtype, device=device)
        self.z = nn.Parameter(z)

    @property
    def device(self) -> torch.device:
        return self.edge_index.device

    def gate_values(self) -> torch.Tensor:
        """S = sigmoid(z / t)"""
        return torch.sigmoid(self.z / self.t)

    # ★ [복구됨] main.py에서 호출하는 메서드
    def current_adjacency(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """정규화 전 S (edge_index, weight(S)) 반환"""
        return self.edge_index, self.gate_values()

    # ★ [복구됨] main.py에서 normalized="row" 요청 시 필요
    def row_normalized_adjacency(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """행 정규화 S_hat = D^{-1} S"""
        row = self.edge_index[0]
        w = self.gate_values()
        
        deg = torch.zeros(self.num_nodes, device=self.device, dtype=w.dtype)
        deg.index_add_(0, row, w)
        deg = deg.clamp_min(1e-12)
        
        # w / deg[row]
        w_hat = w / deg[row]
        return self.edge_index, w_hat

    def sym_normalized_adjacency(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """대칭 정규화 S_hat = D^{-1/2} S D^{-1/2}"""
        row, col = self.edge_index[0], self.edge_index[1]
        w = self.gate_values()
        
        deg = torch.zeros(self.num_nodes, device=self.device, dtype=w.dtype)
        deg.index_add_(0, row, w)
        deg = deg.clamp_min(1e-12)
        d_inv_sqrt = deg.pow(-0.5)
        
        w_hat = w * d_inv_sqrt[row] * d_inv_sqrt[col]
        return self.edge_index, w_hat

    def to_sparse_coo(self, normalized: str = "row") -> torch.Tensor:
        """
        normalized 옵션: "row", "sym", or None/False
        """
        if normalized == "row":
            edge_index, w = self.row_normalized_adjacency()
        elif normalized == "sym":
            edge_index, w = self.sym_normalized_adjacency()
        else:
            edge_index, w = self.current_adjacency()
            
        return torch.sparse_coo_tensor(
            edge_index, w, (self.num_nodes, self.num_nodes), device=self.device
        ).coalesce()
        
        
    @torch.no_grad()
    def add_edges(self, new_edge_index: torch.Tensor, init_probs: torch.Tensor) -> int:
        """
        유니버스(엣지 목록) 확장.
        - new_edge_index: [2, E_new] (long, directed; u->v, v->u 별개)
        - init_probs: (E_new,) in [0,1]  (초기 게이트 확률; 'one'이면 1, 'cosine'이면 cos 값)
        - 반환: 실제 추가된 엣지 수(중복/기존 제외)
        """
        if new_edge_index.numel() == 0:
            return 0

        # dtype/device 정합
        new_edge_index = new_edge_index.to(device=self.device, dtype=torch.long)
        p = init_probs.to(device=self.device, dtype=torch.float32)

        N = self.num_nodes

        # ---- 1) 신규 배치 내 중복 제거 ----
        nu, nv = new_edge_index[0], new_edge_index[1]
        new_keys = nu * N + nv
        uniq_keys, inverse = torch.unique(new_keys, sorted=True, return_inverse=True)

        # 2. 각 유니크 값의 '첫 번째' 등장 인덱스를 구하기 위한 로직
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        # 뒤집어서 scatter_를 수행하면 나중에 덮어씌워지는 값이 '앞쪽' 인덱스가 됨
        inverse, perm = inverse.flip([0]), perm.flip([0])
        uniq_idx = inverse.new_empty(uniq_keys.size(0)).scatter_(0, inverse, perm)
        nu, nv = nu[uniq_idx], nv[uniq_idx]
        p = p[uniq_idx]
        new_keys = nu * N + nv  # 갱신

        # ---- 2) 기존 유니버스 중복 제거 ----
        old_keys = (self.edge_index[0] * N + self.edge_index[1]).to(torch.long)
        old_set = set(old_keys.tolist())
        keep_mask = torch.tensor([int(k.item()) not in old_set for k in new_keys],
                                device=self.device, dtype=torch.bool)
        if not keep_mask.any():
            return 0

        nu, nv = nu[keep_mask], nv[keep_mask]
        p = p[keep_mask]

        # ---- 3) 초기화: z_new = t * logit(p_clamped) ⇒ S = sigmoid(z_new / t) = p_clamped ----
        # 0/1 경계 보정은 torch.logit(..., eps)로 일괄 처리
        z_new = self.t * torch.logit(p, eps=1e-4)

        # ---- 4) concat (buffers & params) ----
        added = torch.stack([nu, nv], dim=0).contiguous()            # [2, E_add]
        self.edge_index = torch.cat([self.edge_index, added], dim=1) # buffer
        self.z = nn.Parameter(torch.cat([self.z.detach(), z_new.detach()], dim=0))

        return added.size(1)


def spmat_to_edge_index(A: sp.spmatrix) -> torch.Tensor:
    if not sp.isspmatrix(A):
        raise TypeError("Expected scipy.sparse matrix")
    A = A.tocoo()
    row = torch.from_numpy(A.row.astype(np.int64))
    col = torch.from_numpy(A.col.astype(np.int64))
    return torch.stack([row, col], dim=0)


import pickle
def _load_user_embeddings_from_pkl(path: str, device: torch.device) -> torch.Tensor:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    # 지원하는 키 자동 탐지
    for k in ["user_embeddings", "user", "U"]:
        if isinstance(obj, dict) and k in obj:
            arr = obj[k]
            break
    else:
        # dict가 아니라 바로 array로 저장된 경우도 지원
        arr = obj
    # torch로 변환
    if isinstance(arr, np.ndarray):
        U = torch.from_numpy(arr)
    elif isinstance(arr, torch.Tensor):
        U = arr
    else:
        raise TypeError(f"Unsupported embedding type: {type(arr)} from {path}")
    return U.to(device=device, dtype=torch.float32)


@torch.no_grad()
def add_toppct_edges_from_embeddings(
    dsm,
    data,
    device: torch.device,
    *,
    emb_path: str,
    top_pct: float,                 # 예: 95, 50, 100, 200, 1000
    weight_mode: str = "cosine",    # 'cosine' | 'one'
) -> int:
    """
    관측/등록 엣지와 자기루프를 제외한 모든 (u,v) 중 코사인 유사도 상위 K개를 추가.
    K = round(base_edges * (top_pct/100)). 'directed' 기준. (a,b)와 (b,a)는 별개 후보.
    
    dsm이 None이면 data.train_social_h_list/t_list/w_list에 직접 추가.
    """
    import pickle, numpy as np, scipy.sparse as sp, torch
    import torch.nn.functional as F

    U = _load_user_embeddings_from_pkl(emb_path, device)
    U = F.normalize(U, dim=1)
    n = int(data.n_users)
    assert U.size(0) == n, f"Embedding rows ({U.size(0)}) != n_users ({n})"

    # 2) 관측/이미 등록 엣지 마스크 + 자기루프
    if dsm is not None:
        # DSM이 있는 경우: DSM의 edge_index 사용
        ei = dsm.edge_index.detach().cpu().numpy()
        base_edges = int(ei.shape[1])   # 'directed' 기준
        obs = sp.coo_matrix((np.ones(base_edges, dtype=np.bool_), (ei[0], ei[1])),
                            shape=(n, n), dtype=np.bool_)
    else:
        # DSM이 없는 경우: data의 소셜 그래프 사용
        if len(data.train_social_h_list) > 0:
            h_arr = np.array(data.train_social_h_list)
            t_arr = np.array(data.train_social_t_list)
            base_edges = len(h_arr)
            obs = sp.coo_matrix((np.ones(base_edges, dtype=np.bool_), (h_arr, t_arr)),
                                shape=(n, n), dtype=np.bool_)
        else:
            base_edges = 0
            obs = sp.coo_matrix((n, n), dtype=np.bool_)
    
    obs.setdiag(True)

    # 3) 전체 코사인 행렬 (float32, O(n^2) 메모리)
    sim = (U @ U.t()).to("cpu")     # torch.Tensor [n,n]
    sim.fill_diagonal_(-1.0)
    if obs.nnz:
        rows = torch.from_numpy(obs.row)
        cols = torch.from_numpy(obs.col)
        sim[rows, cols] = -1.0      # 관측/등록/자기루프 제외

    # 4) 후보 평탄화 후 상위 K 인덱스 선택 (음수 제외)
    flat = sim.view(-1)
    # 제외된 위치는 -1.0이므로 0 미만은 후보 아님
    valid_mask = flat >= 0
    valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
    if valid_idx.numel() == 0:
        print("[add_toppct] no candidates")
        return 0

    K = int(round(base_edges * (float(top_pct) / 100.0)))
    K = max(0, min(K, valid_idx.numel()))
    if K == 0:
        print(f"[add_toppct] K=0 (top_pct={top_pct})")
        return 0

    # 상위 K by 값 (동률은 topk가 임의 타이브레이크)
    cand_vals = flat[valid_idx]
    top_vals, top_pos = torch.topk(cand_vals, k=K, largest=True, sorted=False)
    flat_idx = valid_idx[top_pos]

    # 5) (u,v) 복원
    u_idx = (flat_idx // n).to(torch.long)
    v_idx = (flat_idx %  n).to(torch.long)
    scores = top_vals.to(torch.float32)

    # 6) 엣지 추가
    if dsm is not None:
        # DSM에 추가
        ei_new = torch.stack([u_idx.to(device=dsm.device), v_idx.to(device=dsm.device)], dim=0)
        if weight_mode == "cosine":
            pr_new = scores.to(device=dsm.device)
        elif weight_mode == "one":
            pr_new = torch.ones(K, device=dsm.device, dtype=torch.float32)
        elif weight_mode == "zero":
            pr_new = torch.zeros(K, device=dsm.device, dtype=torch.float32)
        added = dsm.add_edges(ei_new, pr_new)
    else:
        # data에 직접 추가
        u_np = u_idx.cpu().numpy()
        v_np = v_idx.cpu().numpy()
        
        # 기존 엣지와 중복 제거
        existing_edges = set(zip(data.train_social_h_list, data.train_social_t_list))
        new_edges = []
        new_weights = []
        added = 0
        
        for i in range(len(u_np)):
            u, v = int(u_np[i]), int(v_np[i])
            if u != v and (u, v) not in existing_edges:  # 자기루프 제외 및 중복 제외
                new_edges.append((u, v))
                if weight_mode == "cosine":
                    new_weights.append(float(scores[i].item()))
                elif weight_mode == "one":
                    new_weights.append(1.0)
                elif weight_mode == "zero":
                    new_weights.append(0.0)
                added += 1
        
        # data에 추가
        if added > 0:
            new_h = [e[0] for e in new_edges]
            new_t = [e[1] for e in new_edges]
            data.train_social_h_list.extend(new_h)
            data.train_social_t_list.extend(new_t)
            if not hasattr(data, 'train_social_w_list') or data.train_social_w_list is None:
                # 기존 가중치가 없으면 모두 1.0으로 초기화
                data.train_social_w_list = [1.0] * (len(data.train_social_h_list) - len(new_h))
            data.train_social_w_list.extend(new_weights)
    
    print(f"[add_toppct] base_edges={base_edges:,d}, top_pct={top_pct}%, "
          f"request K={K:,d} | added={added:,d} | mean_cos={float(scores.mean()):.6f}")

    return int(added)

