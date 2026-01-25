import pytest
import torch
from torch import nn
from torch.func import jacrev

from bae.autograd.function import TrackingTensor as Track
from bae.autograd.graph import jacobian as sparse_jacobian


class ToyResidual(nn.Module):
    def __init__(self, A: torch.Tensor, B: torch.Tensor):
        super().__init__()
        self.A = nn.Parameter(Track(A))
        self.B = nn.Parameter(Track(B))

    def forward(
        self,
        obs: torch.Tensor,
        idx_a: torch.Tensor,
        idx_b: torch.Tensor,
        sel: torch.Tensor,
    ) -> torch.Tensor:
        a = self.A[idx_a][sel]
        b = self.B[idx_b][sel]
        obs = obs[sel]
        return (a + b) - obs


def _flatten_jac(J: torch.Tensor) -> torch.Tensor:
    n, outdim, num, indim = J.shape
    return J.reshape(n * outdim, num * indim)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sparse_jacobian_matches_torch_jacrev(device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)
    dtype = torch.float64

    num_a, num_b = 4, 5
    n = 7
    dim = 3

    A0 = torch.randn(num_a, dim, device=device, dtype=dtype, requires_grad=True)
    B0 = torch.randn(num_b, dim, device=device, dtype=dtype, requires_grad=True)
    obs = torch.randn(n, dim, device=device, dtype=dtype)

    idx_a = torch.randint(0, num_a, (n,), device=device, dtype=torch.int32)
    idx_b = torch.randint(0, num_b, (n,), device=device, dtype=torch.int32)
    sel = torch.tensor([0, 2, 2, 5, 6], device=device, dtype=torch.int32)

    model = ToyResidual(A0, B0)
    out = model(obs, idx_a, idx_b, sel)

    J_sparse = sparse_jacobian(out, [model.A, model.B])
    assert len(J_sparse) == 2
    assert all(j.layout == torch.sparse_bsr for j in J_sparse)

    def f(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        a = A[idx_a][sel]
        b = B[idx_b][sel]
        obs_sel = obs[sel]
        return (a + b) - obs_sel

    JA, JB = jacrev(f, argnums=(0, 1))(A0, B0)

    torch.testing.assert_close(J_sparse[0].to_dense(), _flatten_jac(JA), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(J_sparse[1].to_dense(), _flatten_jac(JB), rtol=1e-10, atol=1e-10)

    assert torch.equal(J_sparse[0].col_indices(), idx_a[sel])
    assert torch.equal(J_sparse[1].col_indices(), idx_b[sel])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sparse_jacobian_last_op_indexing_is_identity(device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)
    dtype = torch.float64

    num_a = 6
    n = 8
    dim = 4

    A0 = torch.randn(num_a, dim, device=device, dtype=dtype, requires_grad=True)
    idx_a = torch.randint(0, num_a, (n,), device=device, dtype=torch.int32)

    model = nn.Parameter(Track(A0))
    out = model[idx_a]

    (J_sparse,) = sparse_jacobian(out, [model])
    assert J_sparse.layout == torch.sparse_bsr

    def f(A: torch.Tensor) -> torch.Tensor:
        return A[idx_a]

    (JA,) = jacrev(f, argnums=(0,))(A0)
    torch.testing.assert_close(J_sparse.to_dense(), _flatten_jac(JA), rtol=1e-10, atol=1e-10)
    assert torch.equal(J_sparse.col_indices(), idx_a)
