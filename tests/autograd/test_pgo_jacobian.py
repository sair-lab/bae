from __future__ import annotations

from pathlib import Path
import sys

import pypose as pp
import pytest
import torch
from torch.func import jacrev

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bae.autograd.graph import jacobian as sparse_jacobian  # noqa: E402
from bae.utils.ceres_pose import pose_plus_jacobian_xyzw  # noqa: E402
from bae.utils.pgo_dataset import G2OPGO  # noqa: E402
from pgo import (  # noqa: E402
    PoseGraph,
    PoseGraphFixedFirst,
    _ceres_pose_graph_residual,
    _pose_graph_residual,
)


pytestmark = [
    pytest.mark.filterwarnings(r"ignore:CUDA initialization.*:UserWarning"),
    pytest.mark.filterwarnings(r"ignore:Sparse BSR tensor support is in beta state.*:UserWarning"),
]


_PARKING_DATAROOT = _REPO_ROOT / "examples" / "module" / "pgo" / "data"
_PARKING_FILENAME = "parking-garage.g2o"
_JACOBIAN_EDGE_SUBSET = 16


def _dense_residual(
    poses: torch.Tensor,
    node1: torch.Tensor,
    node2: torch.Tensor,
    infos: torch.Tensor,
) -> torch.Tensor:
    return _pose_graph_residual(poses, node1, node2, infos)


def _flatten_jac(jac: torch.Tensor) -> torch.Tensor:
    # [num_edges, 6, num_nodes, 7] -> [num_edges*6, num_nodes*7]
    return jac.reshape(jac.shape[0] * jac.shape[1], jac.shape[2] * jac.shape[3])


def _trim_quat_column(jac_dense: torch.Tensor, num_blocks: int) -> torch.Tensor:
    # Mirror trim_SE3_grad behavior in bae.autograd.graph: drop the last
    # quaternion component in each 7D node block.
    return jac_dense.reshape(jac_dense.shape[0], num_blocks, 7)[..., :6].reshape(jac_dense.shape[0], num_blocks * 6)


def _localize_pose_blocks_ceres(jac_dense: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
    jac_dense = jac_dense.reshape(jac_dense.shape[0], nodes.shape[0], 7)
    plus = pose_plus_jacobian_xyzw(nodes)
    return torch.einsum("bni,nij->bnj", jac_dense, plus).reshape(jac_dense.shape[0], nodes.shape[0] * 6)


def _load_parking_subset(device: torch.device, dtype: torch.dtype):
    path = _PARKING_DATAROOT / _PARKING_FILENAME
    if not path.exists():
        pytest.skip(f"Missing PGO dataset: {path}")

    data = G2OPGO(str(_PARKING_DATAROOT), _PARKING_FILENAME, device=str(device), download=False)
    nodes = data.nodes.tensor() if isinstance(data.nodes, pp.LieTensor) else data.nodes
    poses = data.poses.tensor() if isinstance(data.poses, pp.LieTensor) else data.poses

    edges_full = data.edges[:_JACOBIAN_EDGE_SUBSET].to(torch.long)
    unique_nodes = torch.unique(edges_full.reshape(-1), sorted=True)

    lut = torch.full((int(unique_nodes.max()) + 1,), -1, dtype=torch.long, device=device)
    lut[unique_nodes] = torch.arange(unique_nodes.numel(), dtype=torch.long, device=device)
    edges = lut[edges_full]

    nodes = nodes[unique_nodes].to(dtype=dtype)
    poses = poses[:_JACOBIAN_EDGE_SUBSET].to(dtype=dtype)
    infos = torch.linalg.cholesky(data.infos[:_JACOBIAN_EDGE_SUBSET].to(dtype=dtype))

    return nodes, edges, poses, infos


def test_pose_graph_log_residual_is_zero_at_measurement():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    nodes = pp.randn_SE3(2, sigma=0.2, device=device).tensor().to(dtype=dtype)
    n1 = pp.SE3(nodes[:1])
    n2 = pp.SE3(nodes[1:2])
    q_ab = n1.rotation().Inv() @ n2.rotation()
    p_ab = n1.rotation().Inv() @ (n2.translation() - n1.translation())
    poses = torch.cat((p_ab, q_ab.tensor()), dim=-1)
    infos = torch.eye(6, dtype=dtype, device=device).unsqueeze(0)

    actual = _pose_graph_residual(poses, nodes[:1], nodes[1:2], infos)
    torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=1e-8, atol=1e-8)


def test_ceres_pose_graph_residual_matches_closed_form():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    nodes = pp.randn_SE3(2, sigma=0.2, device=device).tensor().to(dtype=dtype)
    n1 = pp.SE3(nodes[:1])
    n2 = pp.SE3(nodes[1:2])
    poses = pp.randn_SE3(1, sigma=0.1, device=device).tensor().to(dtype=dtype)
    infos = torch.eye(6, dtype=dtype, device=device).unsqueeze(0)

    q_a = n1.rotation()
    p_a = n1.translation()
    q_b = n2.rotation()
    p_b = n2.translation()
    q_ab_est = q_a.Inv() @ q_b
    p_ab_est = q_a.Inv() @ (p_b - p_a)
    delta_q = pp.SO3(poses[:, 3:7]) @ q_ab_est.Inv()
    expected = torch.cat((p_ab_est - poses[:, :3], 2.0 * delta_q.tensor()[..., :3]), dim=-1)

    actual = _ceres_pose_graph_residual(poses, nodes[:1], nodes[1:2], infos)
    torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required (CPU sparse BSR add does not support 6x7 blocks)")
def test_ceres_pgo_jacobian_gauge_free_sparse_matches_dense():
    device = torch.device("cuda")
    dtype = torch.float64

    nodes0, edges, poses, infos = _load_parking_subset(device, dtype)
    model = PoseGraph(nodes0.clone(), residual_type="ceres").to(device)
    residual = model(edges, poses, infos)

    (J_sparse,) = sparse_jacobian(residual, [model.nodes])
    assert J_sparse.layout == torch.sparse_bsr

    def f(nodes: torch.Tensor) -> torch.Tensor:
        return _ceres_pose_graph_residual(poses, nodes[edges[:, 0]], nodes[edges[:, 1]], infos)

    J_dense = jacrev(f)(nodes0)
    J_dense = _localize_pose_blocks_ceres(_flatten_jac(J_dense), nodes0)

    assert J_sparse.shape == J_dense.shape
    torch.testing.assert_close(J_sparse.to_dense(), J_dense, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required (CPU sparse BSR add does not support 6x7 blocks)")
def test_ceres_pgo_jacobian_fixed_first_sparse_matches_dense():
    device = torch.device("cuda")
    dtype = torch.float64

    nodes0, edges, poses, infos = _load_parking_subset(device, dtype)
    assert nodes0.shape[0] >= 2

    node_fixed = nodes0[:1].clone()
    nodes_rest0 = nodes0[1:].clone()
    model = PoseGraphFixedFirst(nodes_rest0.clone(), residual_type="ceres").to(device)
    residual = model(edges, poses, infos, node_fixed)

    (J_sparse,) = sparse_jacobian(residual, [model.nodes_rest])
    assert J_sparse.layout == torch.sparse_bsr

    def f(nodes_rest: torch.Tensor) -> torch.Tensor:
        nodes = torch.cat([node_fixed, nodes_rest], dim=0)
        return _ceres_pose_graph_residual(poses, nodes[edges[:, 0]], nodes[edges[:, 1]], infos)

    J_dense = jacrev(f)(nodes_rest0)
    J_dense = _localize_pose_blocks_ceres(_flatten_jac(J_dense), nodes_rest0)

    assert J_sparse.shape == J_dense.shape
    torch.testing.assert_close(J_sparse.to_dense(), J_dense, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required (CPU sparse BSR add does not support 6x7 blocks)")
def test_pgo_jacobian_gauge_free_sparse_matches_dense():
    device = torch.device("cuda")
    dtype = torch.float64

    nodes0, edges, poses, infos = _load_parking_subset(device, dtype)
    model = PoseGraph(nodes0.clone(), residual_type="log").to(device)
    residual = model(edges, poses, infos)

    (J_sparse,) = sparse_jacobian(residual, [model.nodes])
    assert J_sparse.layout == torch.sparse_bsr

    def f(nodes: torch.Tensor) -> torch.Tensor:
        return _dense_residual(poses, nodes[edges[:, 0]], nodes[edges[:, 1]], infos)

    J_dense = jacrev(f)(nodes0)
    J_dense = _trim_quat_column(_flatten_jac(J_dense), num_blocks=nodes0.shape[0])

    assert J_sparse.shape == J_dense.shape
    torch.testing.assert_close(J_sparse.to_dense(), J_dense, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required (CPU sparse BSR add does not support 6x7 blocks)")
def test_pgo_jacobian_fixed_first_gauge_sparse_matches_dense():
    device = torch.device("cuda")
    dtype = torch.float64

    nodes0, edges, poses, infos = _load_parking_subset(device, dtype)
    assert nodes0.shape[0] >= 2

    node_fixed = nodes0[:1].clone()
    nodes_rest0 = nodes0[1:].clone()
    model = PoseGraphFixedFirst(nodes_rest0.clone(), residual_type="log").to(device)
    residual = model(edges, poses, infos, node_fixed)

    (J_sparse,) = sparse_jacobian(residual, [model.nodes_rest])
    assert J_sparse.layout == torch.sparse_bsr

    def f(nodes_rest: torch.Tensor) -> torch.Tensor:
        nodes = torch.cat([node_fixed, nodes_rest], dim=0)
        return _dense_residual(poses, nodes[edges[:, 0]], nodes[edges[:, 1]], infos)

    J_dense = jacrev(f)(nodes_rest0)
    J_dense = _trim_quat_column(_flatten_jac(J_dense), num_blocks=nodes_rest0.shape[0])

    assert J_sparse.shape == J_dense.shape
    torch.testing.assert_close(J_sparse.to_dense(), J_dense, rtol=1e-10, atol=1e-10)
