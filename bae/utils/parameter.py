import pypose as pp
import torch


def parameter_update_shape(param: torch.Tensor) -> torch.Size:
    if param.ndim == 0:
        return param.shape
    if getattr(param, 'trim_SE3_grad', False):
        return torch.Size((*param.shape[:-1], param.shape[-1] - 1))
    if isinstance(param, pp.LieTensor):
        return torch.Size((*param.shape[:-1], param.ltype.manifold[0]))
    return param.shape


def trim_parameter_jacobian_values(param: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    if param.ndim == 0 or values.shape[-1] != param.shape[-1]:
        return values
    if getattr(param, 'trim_SE3_grad', False):
        return torch.cat([values[..., :6], values[..., 7:]], dim=-1)
    if isinstance(param, pp.LieTensor):
        step_dim = int(param.ltype.manifold[0])
        if step_dim != param.shape[-1]:
            return values[..., :step_dim]
    return values
