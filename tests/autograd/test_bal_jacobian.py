from __future__ import annotations

from pathlib import Path
import bz2
import os
import shutil
import sys
import urllib.error
import urllib.request
from urllib.parse import urljoin

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
_BAL_DATA_URL = "https://grail.cs.washington.edu/projects/bal/"
_BAL_SAMPLES: list[tuple[str, str]] = [
    ("trafalgar", "problem-257-65132-pre"),
    ("dubrovnik", "problem-356-226730-pre"),
    ("ladybug", "problem-1723-156502-pre"),
]


def _candidate_bal_urls(dataset: str, bz2_name: str) -> list[str]:
    base = _BAL_DATA_URL if _BAL_DATA_URL.endswith("/") else (_BAL_DATA_URL + "/")
    prefixes = [
        f"data/{dataset}/",  # matches BAL html link format
        f"{dataset}/",
        f"bal/data/{dataset}/",
        "data/",
        "bal/data/",
        "",
    ]
    urls: list[str] = []
    seen: set[str] = set()
    for prefix in prefixes:
        url = urljoin(base, prefix + bz2_name)
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def _download_url(url: str, dst_path: Path, *, timeout_s: float = 60.0) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "bae-pytest/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp, tmp_path.open("wb") as f:
            shutil.copyfileobj(resp, f)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    os.replace(tmp_path, dst_path)


def _ensure_bal_problem_downloaded(dataset: str, problem_name: str, cache_dir: Path) -> Path:
    problem_name = problem_name.removesuffix(".txt").removesuffix(".bz2").removesuffix(".txt")
    txt_path = cache_dir / f"{problem_name}.txt"
    bz2_path = cache_dir / f"{problem_name}.txt.bz2"

    if txt_path.exists() and txt_path.stat().st_size > 0:
        return txt_path

    if not bz2_path.exists() or bz2_path.stat().st_size == 0:
        bz2_name = bz2_path.name
        last_err: BaseException | None = None
        for url in _candidate_bal_urls(dataset, bz2_name):
            try:
                _download_url(url, bz2_path)
                last_err = None
                break
            except urllib.error.URLError as e:
                last_err = e
        if last_err is not None:
            raise last_err

    tmp_txt = txt_path.with_suffix(".txt.tmp")
    try:
        with bz2.open(bz2_path, "rb") as src, tmp_txt.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    except Exception:
        try:
            tmp_txt.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    os.replace(tmp_txt, txt_path)
    return txt_path


@pytest.fixture(scope="session")
def bal_cache_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    override = os.environ.get("BAE_BAL_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()

    # Prefer the repository's `bal_data/` if it already contains the samples
    # (keeps local development fully offline).
    if _BAL_DATA_DIR.exists() and all((_BAL_DATA_DIR / f"{name}.txt").exists() for _, name in _BAL_SAMPLES):
        return _BAL_DATA_DIR

    return Path(tmp_path_factory.mktemp("bal_data"))


def _load_bal_problem(dataset: str, problem_name: str, cache_dir: Path) -> dict:
    # Allow fully-offline runs when *some* BAL samples are already present in the
    # repository's `bal_data/`, even if others are missing.
    normalized = problem_name.removesuffix(".txt").removesuffix(".bz2").removesuffix(".txt")
    local_txt = _BAL_DATA_DIR / f"{normalized}.txt"
    if local_txt.exists() and local_txt.stat().st_size > 0:
        return read_bal_data(str(local_txt), use_quat=True)

    try:
        path = _ensure_bal_problem_downloaded(dataset, problem_name, cache_dir)
    except Exception as e:
        pytest.skip(f"Could not download BAL sample {dataset}/{problem_name}: {e!r}")
    return read_bal_data(str(path), use_quat=True)


def _jtj_diag_from_bsr(J: torch.Tensor) -> torch.Tensor:
    values = J.values()  # (nnz_blocks, block_rows, block_cols)
    contrib = (values * values).sum(dim=-2)  # (nnz_blocks, block_cols)
    col_blocks = J.col_indices().to(torch.int64)
    num_blocks = J.shape[1] // values.shape[-1]
    diag_blocks = torch.zeros((num_blocks, values.shape[-1]), dtype=values.dtype, device=values.device)
    diag_blocks.index_add_(0, col_blocks, contrib)
    return diag_blocks.flatten()


def _assert_bal_correctness_criteria(
    J_cam: torch.Tensor,
    J_pts: torch.Tensor,
    *,
    camera_idx: torch.Tensor,
    point_idx: torch.Tensor,
    n_cams: int,
    n_pts: int,
) -> None:
    # Correctness criterion 1: no empty block-columns in each BSR Jacobian.
    assert torch.equal(J_cam.col_indices(), camera_idx)
    assert torch.equal(J_pts.col_indices(), point_idx)
    assert torch.unique(J_cam.col_indices()).numel() == n_cams
    assert torch.unique(J_pts.col_indices()).numel() == n_pts

    # Correctness criterion 2: after concatenation, diag(J^T J) is fully occupied.
    diag = torch.cat([_jtj_diag_from_bsr(J_cam), _jtj_diag_from_bsr(J_pts)], dim=0)
    assert (diag > 0).all()


def _remove_camera_and_or_point_appearance(
    camera_idx: torch.Tensor,
    point_idx: torch.Tensor,
    *,
    n_cams: int,
    n_pts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Mutate observation index arrays so at least one camera and/or point ID
    # disappears entirely from observations, creating empty block-columns.
    remove_camera = bool(torch.randint(0, 2, (1,)).item())
    remove_point = bool(torch.randint(0, 2, (1,)).item())
    if not remove_camera and not remove_point:
        remove_camera = True

    camera_idx2 = camera_idx.clone()
    point_idx2 = point_idx.clone()

    if remove_camera:
        if n_cams < 2:
            pytest.skip("BAL sample has <2 cameras; cannot construct removal case.")
        cam_remove = int(torch.randint(0, n_cams, (1,)).item())
        cam_offset = int(torch.randint(1, n_cams, (1,)).item())
        cam_repl = (cam_remove + cam_offset) % n_cams
        camera_idx2[camera_idx2 == cam_remove] = cam_repl
        assert (camera_idx2 == cam_remove).sum().item() == 0

    if remove_point:
        if n_pts < 2:
            pytest.skip("BAL sample has <2 points; cannot construct removal case.")
        pt_remove = int(torch.randint(0, n_pts, (1,)).item())
        pt_offset = int(torch.randint(1, n_pts, (1,)).item())
        pt_repl = (pt_remove + pt_offset) % n_pts
        point_idx2[point_idx2 == pt_remove] = pt_repl
        assert (point_idx2 == pt_remove).sum().item() == 0

    return camera_idx2, point_idx2


@pytest.mark.parametrize(
    ("dataset", "problem_name"),
    _BAL_SAMPLES,
    ids=[f"{ds}.{name}" for ds, name in _BAL_SAMPLES],
)
def test_bal_jacobian_structure_no_empty_columns(
    dataset: str,
    problem_name: str,
    bal_cache_dir: Path,
):
    data = _load_bal_problem(dataset, problem_name, bal_cache_dir)

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

    _assert_bal_correctness_criteria(
        J_cam,
        J_pts,
        camera_idx=camera_idx,
        point_idx=point_idx,
        n_cams=n_cams,
        n_pts=n_pts,
    )


@pytest.mark.parametrize(
    ("dataset", "problem_name"),
    _BAL_SAMPLES,
    ids=[f"{ds}.{name}" for ds, name in _BAL_SAMPLES],
)
def test_bal_jacobian_structure_assert_failed_when_missing_observation_appearance(
    dataset: str,
    problem_name: str,
    bal_cache_dir: Path,
):
    data = _load_bal_problem(dataset, problem_name, bal_cache_dir)

    device = torch.device("cpu")
    dtype = torch.float64

    camera_params = data["camera_params"].to(device=device, dtype=dtype)
    points_3d = data["points_3d"].to(device=device, dtype=dtype)
    points_2d = data["points_2d"].to(device=device, dtype=dtype)
    camera_idx = data["camera_index_of_observations"].to(torch.int32).to(device=device)
    point_idx = data["point_index_of_observations"].to(torch.int32).to(device=device)

    n_cams = int(camera_params.shape[0])
    n_pts = int(points_3d.shape[0])

    camera_idx2, point_idx2 = _remove_camera_and_or_point_appearance(
        camera_idx,
        point_idx,
        n_cams=n_cams,
        n_pts=n_pts,
    )

    model = Reproj(camera_params.clone(), points_3d.clone()).to(device)
    residual = model(points_2d, camera_idx2, point_idx2)
    J_cam, J_pts = autograd_graph.jacobian(residual, [model.pose, model.points_3d])

    with pytest.raises(AssertionError):
        _assert_bal_correctness_criteria(
            J_cam,
            J_pts,
            camera_idx=camera_idx2,
            point_idx=point_idx2,
            n_cams=n_cams,
            n_pts=n_pts,
        )
