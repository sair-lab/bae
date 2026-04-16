import os

import torch

_ENV_FLAG = "BAE_USE_PYPOSE_AMBIENT_GRAD"
_PATCH_INSTALLED = False


def pypose_ambient_grad_enabled() -> bool:
    value = os.environ.get(_ENV_FLAG, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _pm(input: torch.Tensor) -> torch.Tensor:
    return torch.sign(torch.sign(input) * 2 + 1)


def _so3_act_forward(X: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    Xv, Xw = X[..., :3], X[..., 3:]
    uv = torch.linalg.cross(Xv, p, dim=-1)
    uv = uv + uv
    return p + Xw * uv + torch.linalg.cross(Xv, uv, dim=-1)


def _se3_act_forward(X: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return X[..., :3] + _so3_act_forward(X[..., 3:], p)


def _so3_mul_forward(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    Xv, Xw, Yv, Yw = X[..., :3], X[..., 3:], Y[..., :3], Y[..., 3:]
    Zv = Xw * Yv + Xv * Yw + torch.linalg.cross(Xv, Yv, dim=-1)
    Zw = Xw * Yw - (Xv * Yv).sum(dim=-1, keepdim=True)
    return torch.cat([Zv, Zw], dim=-1)


def _se3_mul_forward(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    t = X[..., :3] + _so3_act_forward(X[..., 3:], Y[..., :3])
    q = _so3_mul_forward(X[..., 3:], Y[..., 3:])
    return torch.cat((t, q), dim=-1)


def _so3_inv_forward(X: torch.Tensor) -> torch.Tensor:
    return torch.cat((-X[..., :3], X[..., 3:]), dim=-1)


def _se3_inv_forward(X: torch.Tensor) -> torch.Tensor:
    q_inv = _so3_inv_forward(X[..., 3:])
    t_inv = -_so3_act_forward(q_inv, X[..., :3])
    return torch.cat((t_inv, q_inv), dim=-1)


def _so3_log_forward(input: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(input.dtype).eps
    v, w = input[..., :3], input[..., 3:]
    v_norm = torch.norm(v, 2, dim=-1, keepdim=True)
    w_abs = torch.abs(w)
    v_larger_than_eps = v_norm > eps
    w_larger_than_eps = w_abs > eps
    idx1 = v_larger_than_eps & w_larger_than_eps
    idx2 = v_larger_than_eps & (~w_larger_than_eps)
    idx3 = ~v_larger_than_eps

    factor = torch.zeros_like(v_norm, requires_grad=False)
    factor = factor + idx1 * torch.nan_to_num(2.0 * torch.atan(v_norm / w) / v_norm)
    factor = factor + idx2 * torch.nan_to_num(_pm(w) * torch.pi / v_norm)
    factor = factor + idx3 * torch.nan_to_num(2.0 * (1.0 / w - v_norm * v_norm / (3 * w**3)))
    return factor * v


def _se3_log_forward(input: torch.Tensor) -> torch.Tensor:
    from pypose.lietensor.operation import so3_Jl_inv

    phi = _so3_log_forward(input[..., 3:])
    Jl_inv = so3_Jl_inv(phi)
    tau = (Jl_inv @ input[..., :3].unsqueeze(-1)).squeeze(-1)
    return torch.cat([tau, phi], dim=-1)


def _so3_act4_forward(X: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return torch.cat((_so3_act_forward(X, p[..., :3]), p[..., 3:]), dim=-1)


def _se3_act4_forward(X: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    t = _so3_act_forward(X[..., 3:], p[..., :3]) + X[..., :3] * p[..., 3:]
    return torch.cat((t, p[..., 3:]), dim=-1)


def install_pypose_ambient_grad_monkeypatch() -> bool:
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        return False

    import pypose.lietensor.lietensor as lt
    import pypose.lietensor.operation as op

    def _ambient_so3_log(self, X):
        X = X.tensor() if isinstance(X, lt.LieTensor) else X
        return lt.LieTensor(_so3_log_forward(X), ltype=lt.so3_type)

    def _ambient_se3_log(self, X):
        X = X.tensor() if isinstance(X, lt.LieTensor) else X
        return lt.LieTensor(_se3_log_forward(X), ltype=lt.se3_type)

    def _ambient_so3_act(self, X, p):
        assert not self.on_manifold and isinstance(p, torch.Tensor)
        assert p.shape[-1] == 3 or p.shape[-1] == 4, "Invalid Tensor Dimension"
        X = X.tensor() if isinstance(X, lt.LieTensor) else X
        inputs, out_shape = op.broadcast_inputs(X, p)
        if p.shape[-1] == 3:
            out = _so3_act_forward(*inputs)
        else:
            out = _so3_act4_forward(*inputs)
        dim = -1 if out.nelement() != 0 else p.shape[-1]
        return out.view(out_shape + (dim,))

    def _ambient_se3_act(self, X, p):
        assert not self.on_manifold and isinstance(p, torch.Tensor)
        assert p.shape[-1] == 3 or p.shape[-1] == 4, "Invalid Tensor Dimension"
        X = X.tensor() if isinstance(X, lt.LieTensor) else X
        inputs, out_shape = op.broadcast_inputs(X, p)
        if p.shape[-1] == 3:
            out = _se3_act_forward(*inputs)
        else:
            out = _se3_act4_forward(*inputs)
        dim = -1 if out.nelement() != 0 else p.shape[-1]
        return out.view(out_shape + (dim,))

    def _ambient_so3_mul(self, X, Y):
        X = X.tensor() if isinstance(X, lt.LieTensor) else X
        if not self.on_manifold and isinstance(Y, lt.LieTensor) and not Y.ltype.on_manifold:
            inputs, out_shape = op.broadcast_inputs(X, Y.tensor())
            out = _so3_mul_forward(*inputs)
            dim = -1 if out.nelement() != 0 else X.shape[-1]
            return lt.LieTensor(out.view(out_shape + (dim,)), ltype=lt.SO3_type)
        if not self.on_manifold and isinstance(Y, torch.Tensor):
            return self.Act(X, Y)
        if self.on_manifold:
            return lt.LieTensor(torch.mul(X, Y), ltype=lt.SO3_type)
        raise NotImplementedError("Invalid __mul__ operation")

    def _ambient_so3_inv(self, X):
        X = X.tensor() if isinstance(X, lt.LieTensor) else X
        return lt.LieTensor(_so3_inv_forward(X), ltype=lt.SO3_type)

    def _ambient_se3_mul(self, X, Y):
        X = X.tensor() if isinstance(X, lt.LieTensor) else X
        if not self.on_manifold and isinstance(Y, lt.LieTensor) and not Y.ltype.on_manifold:
            Y = Y.tensor() if hasattr(Y, "ltype") else Y
            inputs, out_shape = op.broadcast_inputs(X, Y)
            out = _se3_mul_forward(*inputs)
            dim = -1 if out.nelement() != 0 else X.shape[-1]
            return lt.LieTensor(out.view(out_shape + (dim,)), ltype=lt.SE3_type)
        if not self.on_manifold and isinstance(Y, torch.Tensor):
            return self.Act(X, Y)
        if self.on_manifold:
            return lt.LieTensor(torch.mul(X, Y), ltype=lt.SE3_type)
        raise NotImplementedError("Invalid __mul__ operation")

    def _ambient_se3_inv(self, X):
        X = X.tensor() if isinstance(X, lt.LieTensor) else X
        return lt.LieTensor(_se3_inv_forward(X), ltype=lt.SE3_type)

    lt.SO3Type.Log = _ambient_so3_log
    lt.SE3Type.Log = _ambient_se3_log
    lt.SO3Type.Act = _ambient_so3_act
    lt.SE3Type.Act = _ambient_se3_act
    lt.SO3Type.Mul = _ambient_so3_mul
    lt.SE3Type.Mul = _ambient_se3_mul
    lt.SO3Type.Inv = _ambient_so3_inv
    lt.SE3Type.Inv = _ambient_se3_inv

    _PATCH_INSTALLED = True
    return True


def maybe_install_pypose_ambient_grad_monkeypatch() -> bool:
    if not pypose_ambient_grad_enabled():
        return False
    return install_pypose_ambient_grad_monkeypatch()
