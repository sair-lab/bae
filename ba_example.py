from time import perf_counter

import pypose as pp
import torch
import torch.nn as nn

from datapipes.bal_loader import get_problem
from bae.autograd.function import TrackingTensor, map_transform
from bae.optim import LM
from bae.utils.ba import rotate_quat
from bae.utils.pysolvers import PCG

TARGET_DATASET = "trafalgar"
TARGET_PROBLEM = "problem-257-65132-pre"
# other options:
# TARGET_DATASET = "ladybug"
# TARGET_PROBLEM = "problem-1723-156502-pre"
# TARGET_DATASET = "dubrovnik"
# TARGET_PROBLEM = "problem-356-226730-pre"

DEVICE = "cuda"
OPTIMIZE_INTRINSICS = True
NUM_CAMERA_PARAMS = 10 if OPTIMIZE_INTRINSICS else 7


@map_transform
def project(points, camera_params):
    projection = rotate_quat(points, camera_params[..., :7])
    projection = -projection[..., :2] / projection[..., [2]]

    f = camera_params[..., [-3]]
    k1 = camera_params[..., [-2]]
    k2 = camera_params[..., [-1]]

    n = torch.sum(projection**2, axis=-1, keepdim=True)
    r = 1 + k1 * n + k2 * n**2
    return projection * r * f


class Residual(nn.Module):
    def __init__(self, camera_params, points):
        super().__init__()
        self.pose = nn.Parameter(TrackingTensor(camera_params))
        self.points = nn.Parameter(TrackingTensor(points))
        self.pose.trim_SE3_grad = True

    def forward(self, observes, cidx, pidx):
        points_proj = project(self.points[pidx], self.pose[cidx])
        return points_proj - observes


def least_square_error(camera_params, points, cidx, pidx, observes):
    model = Residual(camera_params, points)
    loss = model(observes, cidx, pidx)
    return torch.sum(loss**2, dim=-1).mean()


def main():
    dataset = get_problem(TARGET_PROBLEM, TARGET_DATASET)
    print(f"Fetched {TARGET_PROBLEM} from {TARGET_DATASET}")

    dataset = {
        key: value.to(DEVICE)
        for key, value in dataset.items()
        if isinstance(value, torch.Tensor)
    }
    input = {
        "observes": dataset["points_2d"],
        "cidx": dataset["camera_index_of_observations"],
        "pidx": dataset["point_index_of_observations"],
    }

    model = Residual(
        dataset["camera_params"][:, :NUM_CAMERA_PARAMS].clone(),
        dataset["points_3d"].clone(),
    ).to(DEVICE)
    strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5**4)
    solver = PCG(tol=1e-4, maxiter=250)
    optimizer = LM(model, strategy=strategy, solver=solver, reject=30)

    print('Loss:', least_square_error(
        model.pose,
        model.points,
        dataset["camera_index_of_observations"],
        dataset["point_index_of_observations"],
        dataset["points_2d"],
    ).item())

    print("Initial loss", optimizer.model.loss(input, None).item())

    start = perf_counter()
    for idx in range(20):
        loss = optimizer.step(input)
        print("Iteration", idx, "loss", loss.item(), "time", perf_counter() - start)

    torch.cuda.synchronize()
    end = perf_counter()
    print("Time", end - start)

    print('Ending loss:', least_square_error(
        model.pose,
        model.points,
        dataset["camera_index_of_observations"],
        dataset["point_index_of_observations"],
        dataset["points_2d"],
    ).item())


if __name__ == "__main__":
    main()
