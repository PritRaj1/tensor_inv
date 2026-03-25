import pytest
import torch

from tensor_inv.ozaki import ozaki_matmul


def _max_rel_err(a, b):
    nz = b.abs() > 1e-12
    if not nz.any():
        return (a - b).abs().max().item()
    return ((a - b).abs()[nz] / b.abs()[nz]).max().item()


@pytest.mark.parametrize("m,k,n", [(64, 64, 64), (37, 51, 29)])
def test_accuracy_cpu(m, k, n):
    torch.manual_seed(42)
    A = torch.randn(m, k, dtype=torch.float64)
    B = torch.randn(k, n, dtype=torch.float64)
    assert _max_rel_err(ozaki_matmul(A, B), A @ B) < 1e-10


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
@pytest.mark.parametrize("m,k,n", [(128, 128, 128), (37, 51, 29)])
def test_accuracy_cuda(m, k, n):
    torch.manual_seed(42)
    A = torch.randn(m, k, dtype=torch.float64, device="cuda")
    B = torch.randn(k, n, dtype=torch.float64, device="cuda")
    assert _max_rel_err(ozaki_matmul(A, B), A @ B) < 1e-10
