from __future__ import annotations

from pathlib import Path
import os
import sys

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ba_helpers import Reproj  # noqa: E402
import bae.autograd.graph as autograd_graph  # noqa: E402
from datapipes.bal_io import read_bal_data  # noqa: E402


pytestmark = [
    pytest.mark.filterwarnings(r"ignore:CUDA initialization.*:UserWarning"),
    pytest.mark.filterwarnings(r"ignore:Sparse BSR tensor support is in beta state.*:UserWarning"),
]

_BAL_DATA_DIR = _REPO_ROOT / "bal_data"
_BAL_PROBLEM_FILES = sorted(_BAL_DATA_DIR.glob("problem-*-pre.txt"))
if not _BAL_PROBLEM_FILES:
    _BAL_PROBLEM_FILES = [_BAL_DATA_DIR / "problem-257-65132-pre.txt"]


def _load_bal_problem(path: Path) -> dict:
    # Keep this fully offline for CI (no torchdata/HttpReader).
    if not path.exists():
        pytest.skip(f"Missing BAL sample file: {path}")
    return read_bal_data(str(path), use_quat=True)


def _jtj_diag_from_bsr(J: torch.Tensor) -> torch.Tensor:
    values = J.values()  # (nnz_blocks, block_rows, block_cols)
    contrib = (values * values).sum(dim=-2)  # (nnz_blocks, block_cols)
    col_blocks = J.col_indices().to(torch.int64)
    num_blocks = J.shape[1] // values.shape[-1]
    diag_blocks = torch.zeros((num_blocks, values.shape[-1]), dtype=values.dtype, device=values.device)
    diag_blocks.index_add_(0, col_blocks, contrib)
    return diag_blocks.flatten()


@pytest.mark.parametrize(
    "problem_path",
    _BAL_PROBLEM_FILES,
    ids=[p.name for p in _BAL_PROBLEM_FILES],
)
def test_bal_jacobian_structure_no_empty_columns(problem_path: Path, monkeypatch: pytest.MonkeyPatch):
    data = _load_bal_problem(problem_path)

    # CPU-only: CI doesn't have CUDA.
    device = torch.device("cpu")
    dtype = torch.float64

    camera_params = data["camera_params"]
    points_3d = data["points_3d"]
    points_2d = data["points_2d"]
    camera_idx = data["camera_index_of_observations"].to(torch.int32)
    point_idx = data["point_index_of_observations"].to(torch.int32)

    camera_params = camera_params.to(device=device, dtype=dtype)
    points_3d = points_3d.to(device=device, dtype=dtype)
    points_2d = points_2d.to(device=device, dtype=dtype)
    camera_idx = camera_idx.to(device=device)
    point_idx = point_idx.to(device=device)

    model = Reproj(camera_params.clone(), points_3d.clone()).to(device)
    residual = model(points_2d, camera_idx, point_idx)

    J_cam, J_pts = autograd_graph.jacobian(residual, [model.pose, model.points_3d])
    assert J_cam.layout == torch.sparse_bsr
    assert J_pts.layout == torch.sparse_bsr

    n_cams = model.pose.shape[0]
    n_pts = model.points_3d.shape[0]

    # Correctness criterion 1: no empty block-columns in each BSR Jacobian.
    assert torch.equal(J_cam.col_indices(), camera_idx)
    assert torch.equal(J_pts.col_indices(), point_idx)
    assert torch.unique(J_cam.col_indices()).numel() == n_cams
    assert torch.unique(J_pts.col_indices()).numel() == n_pts

    # Correctness criterion 2: after concatenation, diag(J^T J) is fully occupied.
    diag = torch.cat([_jtj_diag_from_bsr(J_cam), _jtj_diag_from_bsr(J_pts)], dim=0)
    assert (diag > 0).all()
