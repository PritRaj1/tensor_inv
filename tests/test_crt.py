import pytest
import torch

from tensor_inv.crt_matmul import crt_matmul


@pytest.fixture
def mats():
    torch.manual_seed(42)
    A = torch.randn(37, 51, dtype=torch.float64)
    B = torch.randn(51, 29, dtype=torch.float64)
    return A, B


@pytest.mark.parametrize(
    "dtype,tol",
    [(torch.float32, 1e-3), (torch.float64, 1e-10)],
)
def test_accuracy(mats, dtype, tol):
    A, B = mats
    A, B = A.to(dtype), B.to(dtype)
    C = crt_matmul(A, B)
    ref = A.double() @ B.double()
    nz = ref.abs() > 1e-12
    rel_err = ((C - ref).abs()[nz] / ref.abs()[nz]).max().item()
    assert rel_err < tol
