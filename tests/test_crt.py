import torch

from tensor_inv.crt_matmul import crt_matmul


def test_accuracy():
    torch.manual_seed(42)
    A = torch.randn(37, 51, dtype=torch.float64)
    B = torch.randn(51, 29, dtype=torch.float64)
    C = crt_matmul(A, B)
    ref = A @ B
    nz = ref.abs() > 1e-12
    rel_err = ((C - ref).abs()[nz] / ref.abs()[nz]).max().item()
    assert rel_err < 1e-10
