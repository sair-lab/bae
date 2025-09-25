import torch
import torch.nn as nn
from bae.autograd.function import TrackingTensor, map_transform
from bae.utils.ba import rotate_euler, rotate_quat

USE_QUATERNIONS = True

@map_transform
def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    if USE_QUATERNIONS:
        points_proj = rotate_quat(points, camera_params[..., :7])
    else:
        points_proj = rotate_euler(points, camera_params[..., 3:6])
        points_proj = points_proj + camera_params[..., :3]
    points_proj = -points_proj[..., :2] / points_proj[..., 2].unsqueeze(-1)
    f = camera_params[..., -3].unsqueeze(-1)
    k1 = camera_params[..., -2].unsqueeze(-1)
    k2 = camera_params[..., -1].unsqueeze(-1)
    
    n = torch.sum(points_proj**2, axis=-1, keepdim=True)
    r = 1 + k1 * n + k2 * n**2
    points_proj = points_proj * r * f

    return points_proj

class Reproj(nn.Module):
    def __init__(self, camera_params, points_3d):
        super().__init__()
        self.pose = nn.Parameter(TrackingTensor(camera_params))
        self.points_3d = nn.Parameter(TrackingTensor(points_3d))
        self.pose.trim_SE3_grad = True

    def forward(self, points_2d, camera_indices, point_indices):
        camera_params = self.pose
        points_3d = self.points_3d

        points_proj = project(points_3d[point_indices], camera_params[camera_indices])
        loss = points_proj - points_2d
        return loss

def least_square_error(camera_params, points_3d, camera_indices, point_indices, points_2d):
    model = Reproj(camera_params, points_3d)
    loss = model(points_2d, camera_indices, point_indices)
    return torch.sum(loss**2, dim=-1).mean()
    return torch.sum(loss**2) / 2