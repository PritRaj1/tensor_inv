import torch

from tensor_inv.lu import lu


def test_lu():
    torch.manual_seed(42)
    n = 128
    A = torch.randn(n, n, dtype=torch.float64)

    P, L, U = lu(A, block_size=32)
    err = (P @ A - L @ U).abs().max().item()
    assert err < 1e-8, f"reconstruction error {err:.2e}"
