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


class CatResidual(nn.Module):
    def __init__(self, A: torch.Tensor, B: torch.Tensor):
        super().__init__()
        self.A = nn.Parameter(Track(A))
        self.B = nn.Parameter(Track(B))

    def forward(
        self,
        obs_a: torch.Tensor,
        obs_b: torch.Tensor,
        idx_a: torch.Tensor,
        idx_b: torch.Tensor,
        mul_a: torch.Tensor,
        mul_b: torch.Tensor,
    ) -> torch.Tensor:
        ra = (self.A[idx_a] - obs_a) * mul_a
        rb = (self.B[idx_b] - obs_b) * mul_b
        return torch.cat([ra, rb], dim=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sparse_jacobian_cat_dim0_matches_torch_jacrev(device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)
    dtype = torch.float64

    num_a, num_b = 5, 7
    n_a, n_b = 4, 6
    dim = 3

    A0 = torch.randn(num_a, dim, device=device, dtype=dtype)
    B0 = torch.randn(num_b, dim, device=device, dtype=dtype)
    obs_a = torch.randn(n_a, dim, device=device, dtype=dtype)
    obs_b = torch.randn(n_b, dim, device=device, dtype=dtype)

    idx_a = torch.randint(0, num_a, (n_a,), device=device, dtype=torch.int32)
    idx_b = torch.randint(0, num_b, (n_b,), device=device, dtype=torch.int32)

    mul_a = torch.rand(n_a, dim, device=device, dtype=dtype) + 0.5
    mul_b = torch.rand(n_b, dim, device=device, dtype=dtype) + 0.5

    model = CatResidual(A0, B0)
    out = model(obs_a, obs_b, idx_a, idx_b, mul_a, mul_b)

    JA_sparse, JB_sparse = sparse_jacobian(out, [model.A, model.B])

    def f(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        ra = (A[idx_a] - obs_a) * mul_a
        rb = (B[idx_b] - obs_b) * mul_b
        return torch.cat([ra, rb], dim=0)

    JA, JB = jacrev(f, argnums=(0, 1))(A0, B0)
    torch.testing.assert_close(JA_sparse.to_dense(), _flatten_jac(JA), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(JB_sparse.to_dense(), _flatten_jac(JB), rtol=1e-10, atol=1e-10)

    assert JA_sparse.crow_indices()[n_a].item() == n_a
    assert JA_sparse.crow_indices()[-1].item() == n_a
    assert torch.equal(JA_sparse.col_indices(), idx_a)

    assert JB_sparse.crow_indices()[n_a].item() == 0
    assert JB_sparse.crow_indices()[-1].item() == n_b
    assert torch.equal(JB_sparse.col_indices(), idx_b)


class CatSubResidual(nn.Module):
    def __init__(self, A: torch.Tensor, B: torch.Tensor):
        super().__init__()
        self.A = nn.Parameter(Track(A))
        self.B = nn.Parameter(Track(B))

    def forward(
        self,
        obs_a: torch.Tensor,
        obs_b: torch.Tensor,
        idx_a: torch.Tensor,
        idx_b: torch.Tensor,
        mul_a: torch.Tensor,
        mul_b: torch.Tensor,
    ) -> torch.Tensor:
        pred = torch.cat([self.A[idx_a] * mul_a, self.B[idx_b] * mul_b], dim=0)
        obs = torch.cat([obs_a, obs_b], dim=0)
        return pred - obs


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sparse_jacobian_cat_minus_cat_matches_torch_jacrev(device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)
    dtype = torch.float64

    num_a, num_b = 6, 8
    n_a, n_b = 5, 7
    dim = 3

    A0 = torch.randn(num_a, dim, device=device, dtype=dtype)
    B0 = torch.randn(num_b, dim, device=device, dtype=dtype)
    obs_a = torch.randn(n_a, dim, device=device, dtype=dtype)
    obs_b = torch.randn(n_b, dim, device=device, dtype=dtype)

    idx_a = torch.randint(0, num_a, (n_a,), device=device, dtype=torch.int32)
    idx_b = torch.randint(0, num_b, (n_b,), device=device, dtype=torch.int32)

    mul_a = torch.rand(n_a, dim, device=device, dtype=dtype) + 0.5
    mul_b = torch.rand(n_b, dim, device=device, dtype=dtype) + 0.5

    model = CatSubResidual(A0, B0)
    out = model(obs_a, obs_b, idx_a, idx_b, mul_a, mul_b)

    JA_sparse, JB_sparse = sparse_jacobian(out, [model.A, model.B])

    def f(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        pred = torch.cat([A[idx_a] * mul_a, B[idx_b] * mul_b], dim=0)
        obs = torch.cat([obs_a, obs_b], dim=0)
        return pred - obs

    JA, JB = jacrev(f, argnums=(0, 1))(A0, B0)
    torch.testing.assert_close(JA_sparse.to_dense(), _flatten_jac(JA), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(JB_sparse.to_dense(), _flatten_jac(JB), rtol=1e-10, atol=1e-10)

    assert JA_sparse.crow_indices()[n_a].item() == n_a
    assert JA_sparse.crow_indices()[-1].item() == n_a
    assert torch.equal(JA_sparse.col_indices(), idx_a)

    assert JB_sparse.crow_indices()[n_a].item() == 0
    assert JB_sparse.crow_indices()[-1].item() == n_b
    assert torch.equal(JB_sparse.col_indices(), idx_b)


class CatIndexResidual(nn.Module):
    def __init__(self, A: torch.Tensor, B: torch.Tensor):
        super().__init__()
        self.A = nn.Parameter(Track(A))
        self.B = nn.Parameter(Track(B))

    def forward(self, obs: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([self.A, self.B], dim=0)
        return cat[idx] - obs


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sparse_jacobian_index_after_cat_matches_torch_jacrev(device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)
    dtype = torch.float64

    num_a, num_b = 4, 6
    dim = 3
    n = 9

    A0 = torch.randn(num_a, dim, device=device, dtype=dtype)
    B0 = torch.randn(num_b, dim, device=device, dtype=dtype)
    obs = torch.randn(n, dim, device=device, dtype=dtype)
    idx = torch.randint(0, num_a + num_b, (n,), device=device, dtype=torch.int32)

    model = CatIndexResidual(A0, B0)
    out = model(obs, idx)
    JA_sparse, JB_sparse = sparse_jacobian(out, [model.A, model.B])

    def f(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([A, B], dim=0)
        return cat[idx] - obs

    JA, JB = jacrev(f, argnums=(0, 1))(A0, B0)
    torch.testing.assert_close(JA_sparse.to_dense(), _flatten_jac(JA), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(JB_sparse.to_dense(), _flatten_jac(JB), rtol=1e-10, atol=1e-10)
