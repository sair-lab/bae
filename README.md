# `bae`: Bundle Adjustment in the Eager-mode

> **⚠️ Development Phase Notice**: This library is currently in active development. APIs are subject to change and should be considered experimental. Use at your own discretion in production environments.

`bae` is a PyTorch-based library supporting 2nd-order optimization techniques. The library provides efficient implementations for sparse optimization problems in robotics, particularly Bundle Adjustment (BA) and Pose Graph Optimization (PGO).

## Features

- **Sparse Block Matrix Operations**: Optimized implementations of sparse matrix operations for large-scale optimization
- **CUDA Acceleration**: Custom CUDA kernels for high-performance sparse linear algebra
- **Bundle Adjustment**: Efficient implementation for camera pose and 3D structure optimization
- **Pose Graph Optimization**: Tools for optimizing robot trajectories using pose graph representations
- **PyTorch Integration**: Seamlessly integrates with PyTorch's automatic differentiation framework
- **Levenberg-Marquardt Optimizer**: Custom implementation of the LM algorithm for non-linear least squares problems

## Installation

### Prerequisites

- CUDA toolkit (tested with CUDA 12.x)
- PyTorch (2.0+)
- (Optional) [CUDSS](https://developer.nvidia.com/cudss) (CUDA Sparse Solver library)

### Setup Instructions

1. (Optional) Install CUDSS (recommended through package manager)
2. Install PyPose from the bae branch:
   ```bash
   pip install git+https://github.com/pypose/pypose.git@bae
   ```
3. Clone this repository:
   ```bash
   git clone https://github.com/zitongzhan/bae.git
   cd bae
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Install the package in development mode:
   ```bash
   pip install -e .
   ```

### Build with CUDSS Tarball (unstable)
If you are unable to install cudss with the system package manager, you can control the build process with these environment variables:

- `USE_CUDSS`: Set to "1" (default) to enable CUDSS support, "0" to disable
- `CUDSS_DIR`: Optional path to CUDSS installation directory if not in standard locations

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

## Dataset Support

The library supports common optimization datasets and tasks:

- **Bundle Adjustment in the Large (BAL)** dataset
- **1DSfM** dataset for large-scale structure from motion
- **G2O** pose graph datasets

## Performance

`bae` is designed for high performance using:

- Efficient sparse block matrix operations
- CUDA acceleration for core operations
- Optimized linear solvers (PCG, CUDA Sparse Solver)
- Memory-efficient data structures

## Citation

If you use `bae` in your research, please cite:

```bibtex
@article{zhan2025bundle,
  title = {Bundle Adjustment in the Eager Mode},
  author = {Zhan, Zitong and Xu, Huan and Fang, Zihang and Wei, Xinpeng and Hu, Yaoyu and Wang, Chen},
  journal = {arXiv preprint arXiv:2409.12190},
  year = {2025},
  url = {https://arxiv.org/abs/2409.12190}
}
```

## Acknowledgements

The implementation draws inspiration from:
- [PyPose](https://github.com/pypose/pypose) for SE(3) pose representations
- GTSAM for reprojection jacobian concepts
