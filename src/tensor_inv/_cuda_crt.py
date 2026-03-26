import math
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_module = None
_CSRC = Path(__file__).parent / "csrc" / "crt_kernel.cu"


def _load():
    global _module
    if _module is None:
        _module = load(
            name="crt_cuda",
            sources=[str(_CSRC)],
            verbose=False,
        )
    return _module


def _crt_weights(moduli):
    """CRT weights as triple-float (hi, mid, lo)"""
    M = math.prod(moduli)
    wh, wm, wl = [], [], []
    for m in moduli:
        Mi = M // m
        w = Mi * pow(Mi, -1, m)
        hi = float(w)
        r = w - int(hi)
        mid = float(r)
        lo = float(r - int(mid))
        wh.append(hi)
        wm.append(mid)
        wl.append(lo)

    M_hi = float(M)
    M_lo = float(M - int(M_hi))
    inv_M = 1.0 / float(M)
    return wh, wm, wl, M_hi, M_lo, inv_M


def cuda_crt_reconstruct(residues, moduli, bits, row_exp, col_exp):
    """CRT reconstruction via CUDA kernel"""
    mod = _load()
    n_mod = len(residues)
    rows, cols = residues[0].shape
    device = residues[0].device

    res_stack = torch.stack(residues).contiguous()
    wh, wm, wl, M_hi, M_lo, inv_M = _crt_weights(moduli)
    wh_t = torch.tensor(wh, dtype=torch.float64, device=device)
    wm_t = torch.tensor(wm, dtype=torch.float64, device=device)
    wl_t = torch.tensor(wl, dtype=torch.float64, device=device)

    return mod.crt_reconstruct(
        res_stack,
        wh_t,
        wm_t,
        wl_t,
        M_hi,
        M_lo,
        inv_M,
        row_exp.int().contiguous(),
        col_exp.int().contiguous(),
        2 * bits,
        rows,
        cols,
        n_mod,
    )
