<picture>
  <img src="https://raw.githubusercontent.com/sair-lab/bae/93eb45121965719c1abbed9437b4832f669b8c87/assets/github-banner.svg" alt="bundle adjustment in the eager-mode" width="100%" />
</picture>

<!-- <p align="center">
  <a href="https://github.com/zitongzhan">Zitong Zhan</a>, <a href="https://www.linkedin.com/in/huan-xu-999700169/?locale=en_US">Huan Xu</a>, Zihang Fang, <a href="https://www.linkedin.com/in/william-xp-wei/">Xinpeng Wei</a>, <a href="https://theairlab.org/team/yaoyuh/">Yaoyu Hu</a>, and <a href="https://sairlab.org">Chen Wang</a>
</p> -->

<p align="center">
  <a href="https://pypose.org/bae/">🌐 Project Page</a> | <a href="https://arxiv.org/abs/2409.12190">📄 PDF</a>
</p>


`bae` is a PyTorch-based library supporting **exact** 2nd-order optimization techniques. The library provides efficient implementations for sparse optimization problems in robotics, particularly Bundle Adjustment (BA) and Pose Graph Optimization (PGO).

### Bundle Adjustment

<table>
  <tr>
    <td align="center" width="33%">
      <p align="center" width="100%">
        <img src="https://github.com/sair-lab/bae/blob/gh-page/docs/assets/garden_half.gif?raw=true" alt="Garden bundle adjustment example" width="100%" />
      </p>
    </td>
    <td align="center" width="33%">
      <p align="center" width="100%">
        <img src="https://github.com/sair-lab/bae/blob/gh-page/docs/assets/counter_half.gif?raw=true" alt="Counter bundle adjustment example" width="100%" />
      </p>
    </td>
    <td align="center" width="33%">
      <p align="center" width="100%">
        <img src="https://github.com/sair-lab/bae/blob/gh-page/docs/assets/kitchen_half.gif?raw=true" alt="Kitchen bundle adjustment example" width="100%" />
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">Garden</td>
    <td align="center">Counter</td>
    <td align="center">Kitchen</td>
  </tr>
</table>

<p align="center"><sub><code>bae</code> powering BA and global positioning in downstream system, <a href="https://github.com/cre185/InstantSfM">InstantSfM</a>.</sub></p>

### Pose Graph Optimization

<table>
  <tr>
    <td align="center" width="33%">
      <img src="https://github.com/sair-lab/bae/blob/gh-page/docs/assets/sphere_bignoise_vertex3.gif?raw=true" alt="Sphere big-noise optimization" width="100%" />
    </td>
    <td align="center" width="33%">
      <img src="https://github.com/sair-lab/bae/blob/gh-page/docs/assets/grid3D.gif?raw=true" alt="3D grid optimization" width="100%" />
    </td>
    <td align="center" width="33%">
      <img src="https://github.com/sair-lab/bae/blob/gh-page/docs/assets/sphere_g2o.gif?raw=true" alt="Sphere g2o optimization" width="100%" />
    </td>
  </tr>
  <tr>
    <td align="center">Sphere Big Noise</td>
    <td align="center">Grid3D</td>
    <td align="center">Sphere (g2o)</td>
  </tr>
</table>

## News

- 2026-03-22: Added [skills](.agent/skills) for coding agents to write custom compute graphs.
- 2025-12-12: Added a VGGT integration example.

## Features

- **Sparse Block Matrix Operations**: Optimized implementations of sparse matrix operations for large-scale optimization
- **CUDA Acceleration**: Custom CUDA kernels for high-performance sparse linear algebra
- **Bundle Adjustment**: Efficient implementation for camera pose and 3D structure optimization
- **Pose Graph Optimization**: Tools for optimizing robot trajectories using pose graph representations
- **PyTorch Integration**: Seamlessly integrates with PyTorch's automatic differentiation framework
- **Levenberg-Marquardt Optimizer**: Custom implementation of the LM algorithm for non-linear least squares problems

### Future Plan
- [ ] Add Apple Silicon GPU support, [PyTorch PR WIP](https://github.com/pytorch/pytorch/pull/177757)
- [ ] Schur complement
- [ ] Reduce runtime overhead using CUDA graph
- [ ] Distributed Tensor (DTensor) support
- [ ] An new backend for [distributed solver](https://github.com/NVIDIA/AMGX)

## Installation

### Prerequisites

- CUDA toolkit (tested with CUDA 12.x)
- PyTorch (2.0+)
- (Optional) [CUDSS](https://developer.nvidia.com/cudss) (CUDA Sparse Solver library)

### Setup Instructions

1. (Optional) Install CUDSS (recommended through package manager)
   - For CUDA 12 (0.6.0)
   ```bash
   sudo apt install cudss=0.6.0-1 cudss0=0.6.0-1 cudss-cuda-12=0.6.0.5-1 \
   libcudss0-cuda-12=0.6.0.5-1 libcudss0-dev-cuda-12=0.6.0.5-1 libcudss0-static-cuda-12=0.6.0.5-1  
   ```
   - For CUDA 13
   ```bash
   sudo apt install cudss-cuda-13
   ```
3. Install PyPose:
   ```bash
   pip install git+https://github.com/pypose/pypose.git
   ```
4. Clone this repository:
   ```bash
   git clone https://github.com/zitongzhan/bae.git
   cd bae
   ```

5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

6. Install the package in development mode:
   ```bash
   python -m pip install --no-build-isolation -v -e .  # following https://github.com/pytorch/pytorch
   ```

### Build with CUDSS Tarball (unstable)
If you are unable to install cudss with the system package manager, you can control the build process with these environment variables:

- `USE_CUDSS`: Set to "1" (default) to enable CUDSS support, "0" to disable
- `CUDSS_DIR`: Optional path to CUDSS installation directory if not in standard locations

## Agent Skills

This repo includes skills in [.agent/skills](.agent/skills):

<!-- - [`bae-codebase`](.agent/skills/bae-codebase/SKILL.md): general guidance for working in this repository -->
- [`bae-compute-graph`](.agent/skills/bae-compute-graph/SKILL.md): guidance for defining BAL/PGO and more complex compute graphs

<!-- Use `bae-compute-graph` for most changes, and add `bae-compute-graph` when working on residual definitions or Jacobian structure. -->

## Example Usage

### Bundle Adjustment

Bundle Adjustment optimizes camera poses and 3D point positions to minimize reprojection error. The following example shows how to perform BA using `bae`:

```python
import torch
import pypose as pp
from datapipes.bal_loader import get_problem
from ba_helpers import ReprojNonBatched, least_square_error
from bae.sparse.py_ops import *
from bae.sparse.solve import *
from bae.optim import LM
from bae.utils.pysolvers import PCG

# Load a problem from the BAL dataset
dataset = get_problem("problem-49-7776-pre", "ladybug", use_quat=True)
dataset = {k: v.to('cuda') for k, v in dataset.items() if isinstance(v, torch.Tensor)}

# Prepare input for the optimization
input = {
    "points_2d": dataset['points_2d'],
    "camera_indices": dataset['camera_index_of_observations'],
    "point_indices": dataset['point_index_of_observations']
}

# Initialize model with camera parameters and 3D points
model = Reproj(
    dataset['camera_params'].clone(),
    dataset['points_3d'].clone()
).to('cuda')

# Configure optimizer
strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5**4)
solver = PCG(tol=1e-4, maxiter=250)
optimizer = LM(model, strategy=strategy, solver=solver, reject=30)

# Run optimization for multiple iterations
for idx in range(20):
    loss = optimizer.step(input)
    print(f'Iteration {idx}, loss: {loss.item()}')
```

### Integration with VGGT

`bae` is used as an optional Bundle Adjustment backend in [our VGGT fork](https://github.com/zitongzhan/vggt) (Visual Geometry Grounded Transformer) to refine the camera poses, intrinsics, and 3D points predicted by VGGT before exporting a COLMAP reconstruction.

After installing `bae`, you can run VGGT's COLMAP export with BA enabled and `bae` selected as the solver:

```bash
python demo_colmap.py --scene_dir /path/to/scene --use_ba --implementation bae  # optional: --shared_camera
```

This command invokes `prepare_bae(...)` inside `vggt/demo_colmap.py`, which wraps VGGT tracks and predictions into `bae.optim.LM` and updates `extrinsic`, `intrinsic`, and `points_3d` in place before writing `scene_dir/sparse/` in COLMAP format.

## Performance

`bae` is designed for high performance using:

- Efficient sparse block matrix operations
- CUDA acceleration for core operations
- Optimized linear solvers (PCG, CUDA Sparse Solver)
- Memory-efficient data structures

## Citation

If you use `bae` in your research, please cite:

```bibtex
@article{zhan2026bundle,
  title = {Bundle Adjustment in the Eager Mode},
  author = {Zhan, Zitong and Xu, Huan and Fang, Zihang and Wei, Xinpeng and Hu, Yaoyu and Wang, Chen},
  journal = {IEEE Transactions on Robotics},
  year = {2026},
  url = {https://arxiv.org/abs/2409.12190}
}
```

## Acknowledgements

The implementation draws inspiration from:
- [PyPose](https://github.com/pypose/pypose) for SE(3) pose representations
- GTSAM for reprojection jacobian concepts
- Ceres for manifold parameter update
