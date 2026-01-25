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


class ToyResidualCat(nn.Module):
    def __init__(self, A: torch.Tensor, B: torch.Tensor):
        super().__init__()
        self.A = nn.Parameter(Track(A))
        self.B = nn.Parameter(Track(B))

    def forward(
        self,
        obs1: torch.Tensor,
        obs2: torch.Tensor,
        idx_a: torch.Tensor,
        idx_b: torch.Tensor,
        sel1: torch.Tensor,
        sel2: torch.Tensor,
    ) -> torch.Tensor:
        a1 = self.A[idx_a][sel1]
        b1 = self.B[idx_b][sel1]
        r1 = (a1 + b1) - obs1[sel1]

        a2 = self.A[idx_a][sel2]
        b2 = self.B[idx_b][sel2]
        r2 = (a2 + b2) - obs2[sel2]
        return torch.cat([r1, r2], dim=0)


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
def test_sparse_jacobian_supports_cat_dim0(device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)
    dtype = torch.float64

    num_a, num_b = 5, 6
    n = 9
    dim = 3

    A0 = torch.randn(num_a, dim, device=device, dtype=dtype, requires_grad=True)
    B0 = torch.randn(num_b, dim, device=device, dtype=dtype, requires_grad=True)
    obs1 = torch.randn(n, dim, device=device, dtype=dtype)
    obs2 = torch.randn(n, dim, device=device, dtype=dtype)

    idx_a = torch.randint(0, num_a, (n,), device=device, dtype=torch.int32)
    idx_b = torch.randint(0, num_b, (n,), device=device, dtype=torch.int32)
    sel1 = torch.tensor([0, 2, 5, 6], device=device, dtype=torch.int32)
    sel2 = torch.tensor([1, 3, 4, 8], device=device, dtype=torch.int32)

    model = ToyResidualCat(A0, B0)
    out = model(obs1, obs2, idx_a, idx_b, sel1, sel2)

    J_sparse = sparse_jacobian(out, [model.A, model.B])
    assert len(J_sparse) == 2
    assert all(j.layout == torch.sparse_bsr for j in J_sparse)

    def f(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        a1 = A[idx_a][sel1]
        b1 = B[idx_b][sel1]
        r1 = (a1 + b1) - obs1[sel1]

        a2 = A[idx_a][sel2]
        b2 = B[idx_b][sel2]
        r2 = (a2 + b2) - obs2[sel2]
        return torch.cat([r1, r2], dim=0)

    JA, JB = jacrev(f, argnums=(0, 1))(A0, B0)

    torch.testing.assert_close(J_sparse[0].to_dense(), _flatten_jac(JA), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(J_sparse[1].to_dense(), _flatten_jac(JB), rtol=1e-10, atol=1e-10)

    assert torch.equal(J_sparse[0].col_indices(), torch.cat([idx_a[sel1], idx_a[sel2]], dim=0))
    assert torch.equal(J_sparse[1].col_indices(), torch.cat([idx_b[sel1], idx_b[sel2]], dim=0))
