from __future__ import annotations

from pathlib import Path
import sys

import pypose as pp
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bae.utils.pypose_ambient_grad import install_pypose_ambient_grad_monkeypatch  # noqa: E402


pytestmark = [
    pytest.mark.filterwarnings(r"ignore:CUDA initialization.*:UserWarning"),
]


def _fd_jacobian(func, x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    y0 = func(x)
    jac = torch.zeros(y0.numel(), x.numel(), dtype=x.dtype, device=x.device)
    for i in range(x.numel()):
        xp = x.clone()
        xm = x.clone()
        xp.view(-1)[i] += eps
        xm.view(-1)[i] -= eps
        jac[:, i] = ((func(xp) - func(xm)) / (2.0 * eps)).reshape(-1)
    return jac.reshape(*y0.shape, *x.shape)


def test_pypose_se3_act_monkeypatch_produces_ambient_gradient():
    install_pypose_ambient_grad_monkeypatch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    pose = pp.randn_SE3(1, sigma=0.2, device=device).tensor().to(dtype=dtype)
    point = torch.randn(1, 3, device=device, dtype=dtype)

    jac = torch.func.jacrev(lambda x: pp.SE3(x).Act(point))(pose)
    jac_fd = _fd_jacobian(lambda x: pp.SE3(x).Act(point), pose)

    torch.testing.assert_close(jac, jac_fd, rtol=1e-7, atol=1e-7)


def test_pypose_se3_mul_monkeypatch_produces_ambient_gradient():
    install_pypose_ambient_grad_monkeypatch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    x = pp.randn_SE3(1, sigma=0.2, device=device).tensor().to(dtype=dtype)
    y = pp.randn_SE3(1, sigma=0.2, device=device).tensor().to(dtype=dtype)

    jac = torch.func.jacrev(lambda a: (pp.SE3(a) * pp.SE3(y)).tensor())(x)
    jac_fd = _fd_jacobian(lambda a: (pp.SE3(a) * pp.SE3(y)).tensor(), x)

    torch.testing.assert_close(jac, jac_fd, rtol=1e-7, atol=1e-7)


def test_pypose_se3_log_monkeypatch_produces_ambient_gradient():
    install_pypose_ambient_grad_monkeypatch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    pose = pp.randn_SE3(1, sigma=0.2, device=device).tensor().to(dtype=dtype)

    jac = torch.func.jacrev(lambda x: pp.SE3(x).Log().tensor())(pose)
    jac_fd = _fd_jacobian(lambda x: pp.SE3(x).Log().tensor(), pose)

    torch.testing.assert_close(jac, jac_fd, rtol=1e-7, atol=1e-7)


def test_pypose_se3_inv_monkeypatch_produces_ambient_gradient():
    install_pypose_ambient_grad_monkeypatch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    pose = pp.randn_SE3(1, sigma=0.2, device=device).tensor().to(dtype=dtype)

    jac = torch.func.jacrev(lambda x: pp.SE3(x).Inv().tensor())(pose)
    jac_fd = _fd_jacobian(lambda x: pp.SE3(x).Inv().tensor(), pose)

    torch.testing.assert_close(jac, jac_fd, rtol=1e-7, atol=1e-7)
