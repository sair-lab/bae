
import torch
import pypose as pp

def rotate_euler(points, rot_vecs):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.
    """
    theta = torch.norm(rot_vecs, dim=-1, keepdim=True)
    v = torch.nan_to_num(rot_vecs / theta)
    dot = torch.sum(points * v, dim=-1, keepdim=True)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    return cos_theta * points + sin_theta * torch.cross(v, points, dim=-1) + dot * (1 - cos_theta) * v

def rotate_quat(points, rot_vecs):
    rot_vecs = pp.SE3(rot_vecs)
    return rot_vecs.Act(points)

# inverse quat
def openGL2gtsam(pose):
    R = pose.rotation()
    t = pose.translation()
    R90 = torch.eye(3, device=pose.device, dtype=pose.dtype)
    R90[0, 0] = 1
    R90[1, 1] = -1
    R90[2, 2] = -1
    wRc = R.Inv() @ pp.mat2SO3(R90)
    t = R.Inv() @ -t
    # // Our camera-to-world translation wTc = -R'*t
    return pp.SE3(torch.cat([t, wRc], dim=-1))
