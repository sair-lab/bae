import pypose as pp
import torch

from .retraction_jacobian import so3_retraction_jacobian, se3_retraction_jacobian
from .pypose_ambient_grad import pypose_ambient_grad_enabled


def parameter_update_shape(param: torch.Tensor) -> torch.Size:
    if param.ndim == 0:
        return param.shape
    if getattr(param, 'trim_SE3_grad', False):
        return torch.Size((*param.shape[:-1], param.shape[-1] - 1))
    if isinstance(param, pp.LieTensor):
        return torch.Size((*param.shape[:-1], param.ltype.manifold[0]))
    return param.shape


def trim_parameter_jacobian_values(
    param: torch.Tensor,
    values: torch.Tensor,
    block_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    if param.ndim == 0 or values.shape[-1] != param.shape[-1]:
        return values
    if getattr(param, 'trim_SE3_grad', False):
        pose = param[..., :7].detach()
        if block_indices is not None:
            pose = pose[block_indices.to(torch.long)]
        pose_values = values[..., :7] @ se3_retraction_jacobian(pose)
        if param.shape[-1] == 7:
            return pose_values
        return torch.cat([pose_values, values[..., 7:]], dim=-1)
    if isinstance(param, pp.LieTensor):
        if pypose_ambient_grad_enabled():
            lie_param = param.detach()
            if block_indices is not None:
                lie_param = lie_param[block_indices.to(torch.long)]
            if param.ltype == pp.SO3_type:
                return values @ so3_retraction_jacobian(lie_param)
            if param.ltype == pp.SE3_type:
                return values @ se3_retraction_jacobian(lie_param)
        step_dim = int(param.ltype.manifold[0])
        if step_dim != param.shape[-1]:
            return values[..., :step_dim]
    return values
