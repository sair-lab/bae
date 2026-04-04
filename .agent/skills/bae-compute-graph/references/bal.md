# BAL Patterns

## Problem structure
- Each observation touches exactly one camera block and one 3D point block.
- The residual is usually 2D reprojection error per observation.
- Sparse layout comes entirely from `camera_indices` and `point_indices`.

## Standard BAL graph

### Parameter setup

```python
class Reproj(nn.Module):
    def __init__(self, camera_params, points_3d):
        super().__init__()
        self.pose = nn.Parameter(TrackingTensor(camera_params))
        self.points_3d = nn.Parameter(TrackingTensor(points_3d))
        self.pose.trim_SE3_grad = True
```

- `self.pose` is typically shape `(num_cameras, 10)` when storing quaternion SE(3) plus 3 intrinsics.
- `trim_SE3_grad = True` means the Jacobian will contain 6 tangent-space pose columns plus any extra Euclidean columns, per parameter block. For 10 camera parameters, gradient becomes 9 optimized columns.
- `self.points_3d` is typically shape `(num_points, 3)`.

### Projection map

```python
@map_transform
def project(points, camera_params):
    points_proj = rotate_quat(points, camera_params[..., :7])
    points_proj = -points_proj[..., :2] / points_proj[..., 2].unsqueeze(-1)

    f = camera_params[..., -3].unsqueeze(-1)
    k1 = camera_params[..., -2].unsqueeze(-1)
    k2 = camera_params[..., -1].unsqueeze(-1)
    n = torch.sum(points_proj ** 2, dim=-1, keepdim=True)
    r = 1 + k1 * n + k2 * n ** 2
    return points_proj * r * f
```

- The function is written with trailing-dimension indexing such as `[..., :2]`, `[..., 2]`, and `dim=-1`, so it works both with and without the extra `vmap` batch wrapper.
- `project(...)` is the `map` op that contributes local Jacobian values.

### Forward pass of `nn.Module`

```python
def forward(self, points_2d, camera_indices, point_indices):
    camera_params = self.pose[camera_indices]
    points = self.points_3d[point_indices]
    pred = project(points, camera_params)
    return pred - points_2d
```

- `self.pose[camera_indices]` defines the camera Jacobian block-columns.
- `self.points_3d[point_indices]` defines the point Jacobian block-columns.
- `pred - points_2d` stays inline because subtraction is already a whitelisted `map` op.

### Expected Jacobians
- `J_cam` has shape `(num_observations * 2, num_cameras * 9)` for quaternion pose plus 3 intrinsics.
- `J_pts` has shape `(num_observations * 2, num_points * 3)`.
- `J_cam.col_indices()` should equal `camera_indices`.
- `J_pts.col_indices()` should equal `point_indices`.

## Gauge-fixed BAL graph

Use this when the first camera pose is fixed and should not appear in the optimized variable set.

### Parameter split

```python
class ReprojFixedFirstCamera(nn.Module):
    def __init__(self, camera_se3_rest, camera_intrinsics, points_3d):
        super().__init__()
        self.pose_rest = nn.Parameter(TrackingTensor(camera_se3_rest))
        self.intrinsics = nn.Parameter(TrackingTensor(camera_intrinsics))
        self.points_3d = nn.Parameter(TrackingTensor(points_3d))
        self.pose_rest.trim_SE3_grad = True
```

- `camera_fixed` is passed into `forward()` as a non-parameter tensor with shape `(1, 7)`.
- `pose_rest` stores only cameras `1..N-1`.
- `intrinsics` still stores all cameras because intrinsics are still optimized for every camera.

### Projection map with split pose/intrinsics

```python
@map_transform
def project_with_se3_and_intrinsics(points, camera_se3, intrinsics):
    points_proj = pp.SE3(camera_se3).Act(points)
    points_proj = -points_proj[..., :2] / points_proj[..., 2].unsqueeze(-1)

    f = intrinsics[..., :1]
    k1 = intrinsics[..., 1:2]
    k2 = intrinsics[..., 2:3]
    n = torch.sum(points_proj ** 2, dim=-1, keepdim=True)
    r = 1 + k1 * n + k2 * n ** 2
    return points_proj * r * f
```

### Forward residual

```python
def forward(self, points_2d, camera_indices, point_indices, camera_fixed):
    camera_se3 = torch.cat([camera_fixed, self.pose_rest], dim=0)
    pred = project_with_se3_and_intrinsics(
        self.points_3d[point_indices],
        camera_se3[camera_indices],
        self.intrinsics[camera_indices],
    )
    return pred - points_2d
```

### Why this works
- `torch.cat([camera_fixed, self.pose_rest], dim=0)` rebuilds the full pose table, but only `self.pose_rest` participates in optimization.
- During backward, `cat(dim=0)` routes Jacobian columns that belong to camera `0` into the fixed tensor branch and routes cameras `1..N-1` into `pose_rest`.
- The resulting `J_cam_rest.col_indices()` should equal `camera_indices[camera_indices > 0] - 1`.
- The intrinsics Jacobian still uses all camera IDs, so `J_intr.col_indices()` should equal `camera_indices`.

## Split-point BAL graph

Use this when one point subset is optimized directly and another subset is produced by transforming a second tracked point set.

### Extra map

```python
@map_transform
def transform_points(points, se3_params):
    return pp.SE3(se3_params).Act(points)
```

### Model structure

```python
class ReprojCat(nn.Module):
    def __init__(self, camera_params, points_b, points_c, se3_c):
        super().__init__()
        self.pose = nn.Parameter(TrackingTensor(camera_params))
        self.points_b = nn.Parameter(TrackingTensor(points_b))
        self.points_c = nn.Parameter(TrackingTensor(points_c))
        self.se3_c = nn.Parameter(TrackingTensor(se3_c))
        self.pose.trim_SE3_grad = True
        self.se3_c.trim_SE3_grad = True

    def forward(self, points_2d, camera_indices, point_indices):
        points_c_world = transform_points(self.points_c, self.se3_c)
        points_all = torch.cat([self.points_b, points_c_world], dim=0)
        pred = project(points_all[point_indices], self.pose[camera_indices])
        return pred - points_2d
```

### Expected Jacobians
- `J_cam`: camera pose and intrinsics
- `J_b`: direct point subset
- `J_c`: transformed point subset
- `J_se3`: SE(3) transform applied to `points_c`

This pattern is useful when a point block is assembled from multiple sources before indexing by observation.

## Structural checks
- Jacobians should be `torch.sparse_bsr`.
- `col_indices()` should match observation connectivity exactly.
- If every camera and point appears in at least one observation, there should be no empty parameter columns.
- A practical check is that the diagonal of the concatenated `J^T J` is strictly positive for every constrained parameter column.
