from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
import sys
from time import perf_counter
import warnings

import pypose as pp
import pytest
import torch

# The PGO reference costs were calibrated with ambient Lie gradients.
os.environ.setdefault("BAE_USE_PYPOSE_AMBIENT_GRAD", "1")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bae.optim import LM  # noqa: E402
from bae.utils.pgo_dataset import G2OPGO  # noqa: E402
from bae.utils.pysolvers import PCG  # noqa: E402
from pgo import PoseGraphFixedFirst  # noqa: E402


pytestmark = [
    pytest.mark.filterwarnings(r"ignore:CUDA initialization.*:UserWarning"),
    pytest.mark.filterwarnings(r"ignore:Sparse BSR tensor support is in beta state.*:UserWarning"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for PGO convergence regression"),
]


_PGO_DATA_DIR = _REPO_ROOT / "examples" / "module" / "pgo" / "data"
_DTYPE = torch.float64
_PGO_STEPS = 80

# These reference costs were generated from Ceres 
_PGO_EXPECTED_FINAL_COSTS: dict[str, float] = {
    "sphere_bignoise_vertex3.g2o": 1478347.8373251208,
    "torus3D.g2o": 22556.189328244574,
    "grid3D.g2o": 42159.50839315247,
}
_PGO_CERES_TOTAL_RUNTIMES_S: dict[str, float] = {
    "sphere_bignoise_vertex3.g2o": 4.880856,
    "torus3D.g2o": 13.917892,
    "grid3D.g2o": 141.117555,
}


def _run_pgo_final_cost_fixed_first(sample_name: str) -> tuple[float, float]:
    device = torch.device("cuda")
    data = G2OPGO(str(_PGO_DATA_DIR), sample_name, device=str(device), download=True)
    nodes = data.nodes
    poses = data.poses

    nodes = nodes.to(dtype=_DTYPE)
    poses = poses.to(dtype=_DTYPE)
    infos = torch.linalg.cholesky(data.infos.to(dtype=_DTYPE))

    if nodes.shape[0] < 2:
        pytest.skip(f"PGO sample has <2 nodes: {sample_name}")

    node_fixed = nodes[:1].clone()
    model = PoseGraphFixedFirst(nodes[1:].clone()).to(device)
    solver = PCG(tol=1e-5)
    strategy = pp.optim.strategy.Adaptive()
    optimizer = LM(model, solver=solver, strategy=strategy, min=1e-10, reject=30)
    input = {
        "edges": data.edges,
        "poses": poses,
        "infos": infos,
        "node_fixed": node_fixed,
    }

    # LM emits per-step diagnostics; keep the regression test output quiet.
    torch.cuda.synchronize(device)
    start = perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(_PGO_STEPS):
            optimizer.step(input=input, weight=infos)
    torch.cuda.synchronize(device)
    runtime_s = perf_counter() - start

    with torch.no_grad():
        residual = model(data.edges, poses, infos, node_fixed)
        return 0.5 * residual.square().sum().item(), runtime_s


@pytest.mark.parametrize(
    "sample_name",
    [
        "sphere_bignoise_vertex3.g2o",
        "torus3D.g2o",
        "grid3D.g2o",
    ],
)
def test_pgo_convergence_matches_ceres_reference(sample_name: str):
    final_cost, runtime_s = _run_pgo_final_cost_fixed_first(sample_name)
    assert torch.isfinite(torch.tensor(final_cost))

    expected = _PGO_EXPECTED_FINAL_COSTS.get(sample_name)
    assert expected is not None
    assert final_cost == pytest.approx(expected, abs=2.0)

    ceres_runtime_s = _PGO_CERES_TOTAL_RUNTIMES_S.get(sample_name)
    assert ceres_runtime_s is not None
    speedup_ratio = ceres_runtime_s / runtime_s
    print(
        f"{sample_name}: ours={runtime_s:.6f}s, ceres={ceres_runtime_s:.6f}s, "
        f"speedup_vs_ceres={speedup_ratio:.3f}x"
    )
    if speedup_ratio < 1.0:
        warnings.warn(
            f"{sample_name}: ours is slower than Ceres "
            f"({runtime_s:.6f}s vs {ceres_runtime_s:.6f}s, speedup {speedup_ratio:.3f}x)",
            stacklevel=2,
        )
