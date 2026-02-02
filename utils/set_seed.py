import random
import numpy as np
import torch
try:
    from numba import njit
except Exception:
    njit = None

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True
    print("set pytorch seed:", seed)

if njit is not None:
    @njit
    def _seed_numba(seed):
        np.random.seed(seed)
else:
    _seed_numba = None

def set_numba_seed(seed):
    if _seed_numba is None:
        return
    _seed_numba(seed)
