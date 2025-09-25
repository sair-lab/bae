from time import perf_counter
import torch
import pypose as pp

from ba_helpers import Reproj, least_square_error
from datapipes.bal_loader import get_problem, read_bal_data
from bae.sparse.py_ops import *
from bae.sparse.solve import *
from bae.optim import LM
from bae.utils.pysolvers import PCG, cuSolverSP

# TARGET_DATASET = "ladybug"
# TARGET_PROBLEM = "problem-1723-156502-pre"
# TARGET_PROBLEM = "problem-49-7776-pre"
# TARGET_PROBLEM = "problem-1695-155710-pre"  
# TARGET_PROBLEM = "problem-969-105826-pre"
TARGET_DATASET = "trafalgar"
TARGET_PROBLEM = "problem-257-65132-pre"
# TARGET_DATASET = "dubrovnik"
# TARGET_PROBLEM = "problem-356-226730-pre"



DEVICE = 'cuda'
OPTIMIZE_INTRINSICS = True

USE_QUATERNIONS = True

file_name = f'{TARGET_DATASET}.{TARGET_PROBLEM}'
dataset = get_problem(TARGET_PROBLEM, TARGET_DATASET, use_quat=USE_QUATERNIONS)

if OPTIMIZE_INTRINSICS:
    NUM_CAMERA_PARAMS = 10 if USE_QUATERNIONS else 9
else:
    NUM_CAMERA_PARAMS = 7 if USE_QUATERNIONS else 6

print(f'Fetched {TARGET_PROBLEM} from {TARGET_DATASET}')

trimmed_dataset = dataset
trimmed_dataset = {k: v.to(DEVICE) for k, v in trimmed_dataset.items() if type(v) == torch.Tensor}

input = {
    "points_2d": trimmed_dataset['points_2d'],
    "camera_indices": trimmed_dataset['camera_index_of_observations'],
    "point_indices": trimmed_dataset['point_index_of_observations']
}

model = Reproj(
    trimmed_dataset['camera_params'][:, :NUM_CAMERA_PARAMS].clone(),
    trimmed_dataset['points_3d'].clone()
).to(DEVICE)
strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5**4)
solver = PCG(tol=1e-4, maxiter=250)
optimizer = LM(model, strategy=strategy, solver=solver, reject=30)

print('Loss:', least_square_error(
    model.pose,
    model.points_3d,
    trimmed_dataset['camera_index_of_observations'],
    trimmed_dataset['point_index_of_observations'],
    trimmed_dataset['points_2d'],
).item())

print("Initial loss", optimizer.model.loss(input, None).item())

start = perf_counter()
for idx in range(20):
    loss = optimizer.step(input)
    print('Iteration', idx, 'loss', loss.item(), 'time', perf_counter() - start)

torch.cuda.synchronize()
end = perf_counter()
print('Time', end - start)

print('Ending loss:', least_square_error(
    model.pose,
    model.points_3d,
    trimmed_dataset['camera_index_of_observations'],
    trimmed_dataset['point_index_of_observations'],
    trimmed_dataset['points_2d'],
).item())
