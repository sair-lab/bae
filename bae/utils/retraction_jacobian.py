import torch


def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.unbind(dim=-1)
    zero = torch.zeros_like(x)
    rows = (
        torch.stack((zero, -z, y), dim=-1),
        torch.stack((z, zero, -x), dim=-1),
        torch.stack((-y, x, zero), dim=-1),
    )
    return torch.stack(rows, dim=-2)


def quat_mul_xyzw(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = q1.unbind(dim=-1)
    x2, y2, z2, w2 = q2.unbind(dim=-1)
    return torch.stack((
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ), dim=-1)


def quat_conj_xyzw(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((-q[..., :3], q[..., 3:4]), dim=-1)


def quat_inv_xyzw(q: torch.Tensor) -> torch.Tensor:
    return quat_conj_xyzw(q) / q.square().sum(dim=-1, keepdim=True)


def quat_rotate_xyzw(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qv = torch.cat((v, torch.zeros_like(v[..., :1])), dim=-1)
    return quat_mul_xyzw(quat_mul_xyzw(q, qv), quat_inv_xyzw(q))[..., :3]


def so3_retraction_jacobian(quat: torch.Tensor) -> torch.Tensor:
    qx, qy, qz, qw = quat.unbind(dim=-1)
    rows = (
        torch.stack((qw, qz, -qy), dim=-1),
        torch.stack((-qz, qw, qx), dim=-1),
        torch.stack((qy, -qx, qw), dim=-1),
        torch.stack((-qx, -qy, -qz), dim=-1),
    )
    return 0.5 * torch.stack(rows, dim=-2)


def se3_retraction_jacobian(pose: torch.Tensor) -> torch.Tensor:
    if pose.shape[-1] != 7:
        raise ValueError(f"Expected 7D pose blocks, got {pose.shape[-1]}.")

    J = torch.zeros(*pose.shape[:-1], 7, 6, dtype=pose.dtype, device=pose.device)
    eye = torch.eye(3, dtype=pose.dtype, device=pose.device)
    J[..., :3, :3] = eye
    J[..., :3, 3:6] = -skew_symmetric(pose[..., :3])
    J[..., 3:7, 3:6] = so3_retraction_jacobian(pose[..., 3:7])
    return J
