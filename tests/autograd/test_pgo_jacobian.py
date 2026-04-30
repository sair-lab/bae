from __future__ import annotations

from pathlib import Path
import os
import sys

import pypose as pp
import pytest
import torch
from torch.func import jacrev

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("BAE_USE_PYPOSE_AMBIENT_GRAD", "1")

from bae.autograd.graph import jacobian as sparse_jacobian  # noqa: E402
from bae.utils.retraction_jacobian import se3_retraction_jacobian  # noqa: E402
from bae.utils.pgo_dataset import G2OPGO  # noqa: E402
from pgo import PoseGraph, PoseGraphFixedFirst, _pose_graph_residual  # noqa: E402


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


def _localize_pose_blocks_se3(jac_dense: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
    jac_dense = jac_dense.reshape(jac_dense.shape[0], nodes.shape[0], 7)
    plus = se3_retraction_jacobian(nodes)
    return torch.einsum("bni,nij->bnj", jac_dense, plus).reshape(jac_dense.shape[0], nodes.shape[0] * 6)


def _load_parking_subset(device: torch.device, dtype: torch.dtype):
    data = G2OPGO(str(_PARKING_DATAROOT), _PARKING_FILENAME, device=str(device), download=True)
    nodes = data.nodes if isinstance(data.nodes, pp.LieTensor) else pp.SE3(data.nodes)
    poses = data.poses if isinstance(data.poses, pp.LieTensor) else pp.SE3(data.poses)

    edges_full = data.edges[:_JACOBIAN_EDGE_SUBSET].to(torch.long)
    unique_nodes = torch.unique(edges_full.reshape(-1), sorted=True)

    lut = torch.full((int(unique_nodes.max()) + 1,), -1, dtype=torch.long, device=device)
    lut[unique_nodes] = torch.arange(unique_nodes.numel(), dtype=torch.long, device=device)
    edges = lut[edges_full]

    nodes = nodes[unique_nodes].to(dtype=dtype)
    poses = poses[:_JACOBIAN_EDGE_SUBSET].to(dtype=dtype)
    infos = torch.linalg.cholesky(data.infos[:_JACOBIAN_EDGE_SUBSET].to(dtype=dtype))

    return nodes, edges, poses, infos


def test_pose_graph_residual_matches_ceres_closed_form():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    nodes = pp.randn_SE3(2, sigma=0.2, device=device).to(dtype=dtype)
    n1 = nodes[:1]
    n2 = nodes[1:2]
    poses = pp.randn_SE3(1, sigma=0.1, device=device).to(dtype=dtype)
    infos = torch.eye(6, dtype=dtype, device=device).unsqueeze(0)

    q_a = n1.rotation()
    p_a = n1.translation()
    q_b = n2.rotation()
    p_b = n2.translation()
    q_ab_est = q_a.Inv() @ q_b
    p_ab_est = q_a.Inv() @ (p_b - p_a)
    delta_q = pp.SO3(poses[:, 3:7]) @ q_ab_est.Inv()
    expected = torch.cat((p_ab_est - poses[:, :3], 2.0 * delta_q.tensor()[..., :3]), dim=-1)

    actual = _pose_graph_residual(poses, nodes[:1], nodes[1:2], infos)
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-7)


def test_se3_pose_plus_jacobian_matches_finite_difference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    pose = pp.randn_SE3(sigma=0.2, device=device).to(dtype=dtype)
    plus = se3_retraction_jacobian(pose.tensor())

    eps = 1e-7
    fd = torch.zeros_like(plus)
    for i in range(6):
        delta = torch.zeros(6, dtype=dtype, device=device)
        delta[i] = eps
        forward = (pp.se3(delta).Exp() * pose).tensor()
        backward = (pp.se3(-delta).Exp() * pose).tensor()
        fd[:, i] = (forward - backward) / (2.0 * eps)

    torch.testing.assert_close(plus, fd, rtol=1e-8, atol=1e-8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required (CPU sparse BSR add does not support 6x7 blocks)")
def test_pgo_jacobian_gauge_free_sparse_matches_dense():
    device = torch.device("cuda")
    dtype = torch.float64

    nodes0, edges, poses, infos = _load_parking_subset(device, dtype)
    model = PoseGraph(nodes0.clone()).to(device)
    residual = model(edges, poses, infos)

    (J_sparse,) = sparse_jacobian(residual, [model.nodes])
    assert J_sparse.layout == torch.sparse_bsr

    def f(nodes: torch.Tensor) -> torch.Tensor:
        return _pose_graph_residual(poses, pp.SE3(nodes)[edges[:, 0]], pp.SE3(nodes)[edges[:, 1]], infos)

    J_dense = jacrev(f)(nodes0.tensor())
    J_dense = _localize_pose_blocks_se3(_flatten_jac(J_dense), nodes0.tensor())

    assert J_sparse.shape == J_dense.shape
    torch.testing.assert_close(J_sparse.to_dense(), J_dense, rtol=1e-10, atol=1e-10)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required (CPU sparse BSR add does not support 6x7 blocks)")
def test_pgo_jacobian_fixed_first_sparse_matches_dense():
    device = torch.device("cuda")
    dtype = torch.float64

    nodes0, edges, poses, infos = _load_parking_subset(device, dtype)
    assert nodes0.shape[0] >= 2

    node_fixed = nodes0[:1].clone()
    nodes_rest0 = nodes0[1:].clone()
    model = PoseGraphFixedFirst(nodes_rest0.clone()).to(device)
    residual = model(edges, poses, infos, node_fixed)

    (J_sparse,) = sparse_jacobian(residual, [model.nodes_rest])
    assert J_sparse.layout == torch.sparse_bsr

    def f(nodes_rest: torch.Tensor) -> torch.Tensor:
        nodes = torch.cat([node_fixed, nodes_rest], dim=0)
        return _pose_graph_residual(poses, pp.SE3(nodes)[edges[:, 0]], pp.SE3(nodes)[edges[:, 1]], infos)

    J_dense = jacrev(f)(nodes_rest0.tensor())
    J_dense = _localize_pose_blocks_se3(_flatten_jac(J_dense), nodes_rest0.tensor())

    assert J_sparse.shape == J_dense.shape
    torch.testing.assert_close(J_sparse.to_dense(), J_dense, rtol=1e-10, atol=1e-10)
