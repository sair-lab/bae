import pytest
import pypose as pp
import torch
from torch import nn
from torch.func import jacrev

from bae.autograd.function import TrackingTensor as Track, map_transform
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


@map_transform
def _relative_se3_residual(poses: pp.LieTensor, node1: pp.LieTensor, node2: pp.LieTensor) -> torch.Tensor:
    return (poses.Inv() @ node1.Inv() @ node2).Log().tensor()


@map_transform
def _cat_inside_map(points: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    u = points[..., :1] * scale[..., :1]
    v = points[..., 1:2] * scale[..., 1:2]
    return torch.cat((u, v), dim=-1)


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


class MapCatResidual(nn.Module):
    def __init__(self, points: torch.Tensor):
        super().__init__()
        self.points = nn.Parameter(Track(points))

    def forward(
        self,
        obs: torch.Tensor,
        idx: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        return _cat_inside_map(self.points[idx], scale) - obs


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sparse_jacobian_map_transform_treats_inner_cat_as_opaque(device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)
    dtype = torch.float64

    num_points = 6
    n = 8
    dim = 3

    points0 = torch.randn(num_points, dim, device=device, dtype=dtype)
    idx = torch.randint(0, num_points, (n,), device=device, dtype=torch.int32)
    scale = torch.rand(n, 2, device=device, dtype=dtype) + 0.5
    obs = torch.randn(n, 2, device=device, dtype=dtype)

    model = MapCatResidual(points0)
    out = model(obs, idx, scale)

    (J_sparse,) = sparse_jacobian(out, [model.points])

    def f(points: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                points[idx, :1] * scale[..., :1],
                points[idx, 1:2] * scale[..., 1:2],
            ),
            dim=-1,
        ) - obs

    (J_points,) = jacrev(f, argnums=(0,))(points0)
    torch.testing.assert_close(J_sparse.to_dense(), _flatten_jac(J_points), rtol=1e-10, atol=1e-10)
    assert torch.equal(J_sparse.col_indices(), idx)


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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tracking_lie_tensor_index_and_cat_preserve_ltype(device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)
    dtype = torch.float64

    nodes = nn.Parameter(Track(pp.randn_SE3(5, device=device, dtype=dtype)))
    idx_a = torch.tensor([0, 2, 4], device=device, dtype=torch.int64)
    idx_b = torch.tensor([1, 3, 4], device=device, dtype=torch.int64)

    node_a = nodes[idx_a]
    node_b = nodes[idx_b]
    cat = torch.cat([node_a, node_b], dim=0)

    assert isinstance(nodes, Track)
    assert isinstance(nodes, pp.LieTensor)
    assert isinstance(node_a, Track)
    assert isinstance(node_a, pp.LieTensor)
    assert type(node_a.ltype) is type(nodes.ltype)
    assert hasattr(node_a, "optrace")
    assert node_a.optrace[id(node_a)][0] == "index"

    assert node_a.tensor().shape == (idx_a.numel(), 7)
    assert node_a.translation().shape == (idx_a.numel(), 3)
    assert node_a.rotation().shape == (idx_a.numel(), 4)
    assert isinstance(node_a.Inv(), pp.LieTensor)
    assert isinstance(node_a.Log(), pp.LieTensor)

    assert isinstance(cat, Track)
    assert isinstance(cat, pp.LieTensor)
    assert type(cat.ltype) is type(nodes.ltype)
    assert hasattr(cat, "optrace")
    assert cat.optrace[id(cat)][0] == "cat"
    assert cat.translation().shape == (idx_a.numel() + idx_b.numel(), 3)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sparse_jacobian_matches_lie_tensor_pgo_residual(device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)
    dtype = torch.float64

    num_nodes = 5
    num_edges = 4
    nodes0 = pp.randn_SE3(num_nodes, device=device, dtype=dtype)
    poses = pp.randn_SE3(num_edges, device=device, dtype=dtype)
    idx1 = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int64)
    idx2 = torch.tensor([1, 2, 3, 4], device=device, dtype=torch.int64)

    model = nn.Parameter(Track(nodes0))
    out = _relative_se3_residual(poses, model[idx1], model[idx2])

    (J_sparse,) = sparse_jacobian(out, [model])

    nodes_tensor = nodes0.tensor().detach().clone().requires_grad_(True)

    def f(nodes_data: torch.Tensor) -> torch.Tensor:
        nodes = pp.SE3(nodes_data)
        return (poses.Inv() @ nodes[idx1].Inv() @ nodes[idx2]).Log().tensor()

    (J_dense,) = jacrev(f, argnums=(0,))(nodes_tensor)
    torch.testing.assert_close(J_sparse.to_dense(), _flatten_jac(J_dense[..., :6]), rtol=1e-10, atol=1e-10)
