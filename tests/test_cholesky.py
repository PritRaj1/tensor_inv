import torch

from tensor_inv.cholesky import cholesky


def test_cholesky():
    torch.manual_seed(42)
    n = 128
    A = torch.randn(n, n, dtype=torch.float64)
    A = A @ A.T + n * torch.eye(n, dtype=torch.float64)  # SPD

    L = cholesky(A, block_size=32)
    err = (L @ L.T - A).abs().max().item()
    assert err < 1e-8, f"reconstruction error {err:.2e}"
    assert (torch.diag(L) > 0).all(), "diagonal should be positive"
