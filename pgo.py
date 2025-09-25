from functools import partial
import os
import torch
import argparse
import pypose as pp
from torch import nn
from torch.func import jacrev
from bae.autograd.function import TrackingTensor, map_transform
from bae.autograd.graph import construct_sbt
from bae.sparse.py_ops import diagonal_op_

from bae.utils.pgo_dataset import G2OPGO
from bae.utils.pgo import plot_and_save
import pypose.optim.solver as ppos
import pypose.optim.kernel as ppok
import pypose.optim.corrector as ppoc
import pypose.optim.strategy as ppost
from pypose.optim.scheduler import StopOnPlateau
from bae.utils.pysolvers import PCG, cuSolverSP
from bae.optim import LM

OPTIMIZE_INTRINSICS = False
USE_QUATERNIONS=True
DTYPE = torch.float64

torch.set_printoptions(precision=6)

def diff(residual=None, jacobian=None):
    num_factors = residual.shape[0] if residual is not None else jacobian.shape[0]
    import numpy as np
    with open('data.s', 'r') as f:
        ceres_residuals = []
        ceres_jacobians = []
        for i in range(num_factors):
        # read from 'data.s'
            data = f.readline()
            discard_left = data.split('[')[1:]
            discard_right = [x.split(']')[0] for x in discard_left]
            discard_semi = [x.split(';') for x in discard_right]
            # convert to float
            ceres_residual = [float(y[0]) for y in discard_semi]
            ceres_residuals.append(ceres_residual)
            
            ceres_jacobian = [np.fromstring(y[1], sep=',') for y in discard_semi]
            ceres_jacobians.append(ceres_jacobian)
    ceres_residuals = torch.tensor(ceres_residuals)
    ceres_jacobians = torch.tensor(ceres_jacobians)
    if residual is not None:
        ceres_residuals = ceres_residuals - residual
        # absolute difference
        print(ceres_residuals.norm(dim=-1).mean())
        # relative difference
        print(((ceres_residuals.norm(dim=-1) / residual.norm(dim=-1)))[1:].mean())
    if jacobian is not None:
        ceres_jacobians = ceres_jacobians - jacobian
        # absolute difference

def write_ceres_txt(nodes, filename='data.s'):
    with open(filename, 'w') as f:
        # ID x y z q_x q_y q_z q_w
        for i in range(nodes.shape[0]):
            node = nodes[i]
            f.write(f'{i} {node[0].item()} {node[1].item()} {node[2].item()} {node[3].item()} {node[4].item()} {node[5].item()} {node[6].item()}\n')

def foo(poses, node1, node2, infos):
    node1 = pp.SE3(node1)
    node2 = pp.SE3(node2)
    poses = pp.SE3(poses)
    # The measured relative transform (Pose) is poses = z_ab = [hat{p}_{ab}, hat{q}_{ab}].
    # The predicted relative transform x_ab_est = x_a^-1 @ x_b
    q_a = node1.rotation()
    p_a = node1.translation()
    q_b = node2.rotation()
    p_b = node2.translation()

    q_ab_est = q_a.Inv() @ q_b
    p_ab_est = q_a.Inv() @ (p_b - p_a)

    p_ab_meas = poses.translation()     # hat{p}_{ab}
    q_ab_meas = poses.rotation()        # hat{q}_{ab}

    # Compute the position part of the residual: p_ab_est - p_ab_meas
    r_p = p_ab_est - p_ab_meas

    # Compute the orientation part:
    #   2 * vec( (q_ab_est) * (q_ab_meas^-1) )
    # where q_ab_est * q_ab_meas^-1 is a quaternion, and vec(...) is its imaginary part.
    delta_q = q_ab_meas @ q_ab_est.Inv()
    # PyPose quaternions by default have layout [w, x, y, z];
    # the imaginary part is q_res[..., 1:4].
    r_q = delta_q.tensor()[..., :3]

    # Concatenate into a 6D residual per edge: [ position_error | orientation_error ]
    residual = torch.cat([r_p, r_q], dim=-1)
    residual = infos @ residual[..., None]
    residual = residual[..., 0]
    return residual

@map_transform
def foo(poses, node1, node2, infos):
    residual = (pp.SE3(poses).Inv() @ pp.SE3(node1).Inv() @ pp.SE3(node2)).Log().tensor()
    residual = infos @ residual[..., None]
    residual = residual[..., 0]
    return residual

class PoseGraph(nn.Module):

    def __init__(self, nodes):
        super().__init__()
        self.nodes = nn.Parameter(TrackingTensor(nodes))
        self.nodes.trim_SE3_grad = True

    def forward(self, edges, poses, infos):
        node1 = self.nodes[edges[..., 0]]
        node2 = self.nodes[edges[..., 1]]
        return foo(poses, node1, node2, infos)


from bae.optim import LM


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pose Graph Optimization')
    parser.add_argument("--device", type=str, default='cuda', help="cuda or cpu")
    parser.add_argument("--radius", type=float, default=1e4, help="trust region radius")
    parser.add_argument("--save", type=str, default='./examples/module/pgo/save/', \
                        help="files location to save")
    parser.add_argument("--dataroot", type=str, default='./examples/module/pgo/data', \
                        help="dataset location")
    parser.add_argument("--dataname", type=str, default='parking-garage.g2o', \
                        help="dataset name")  # sphere_bignoise_vertex3, torus3D, grid3D, parking-garage
    parser.add_argument('--no-vectorize', dest='vectorize', action='store_false', \
                        help="to save memory")
    parser.add_argument('--vectorize', action='store_true', \
                        help='to accelerate computation')
    parser.set_defaults(vectorize=True)
    args = parser.parse_args(); print(args)
    os.makedirs(os.path.join(args.save), exist_ok=True)

    data = G2OPGO(args.dataroot, args.dataname, device=args.device, download=True)
    data.nodes = data.nodes.to(DTYPE)
    data.poses = data.poses.to(DTYPE)
    data.infos = data.infos.to(DTYPE)

    edges, poses, infos = data.edges, data.poses, data.infos
    infos = torch.linalg.cholesky(infos)
    input = {'edges': edges, 'poses': poses, 'infos': infos}

    graph = PoseGraph(data.nodes).to(args.device)
    # solver = PCG(tol=1e-5)
    solver = cuSolverSP()
    # solver = ppos.Cholesky()
    # strategy = ppost.TrustRegion(radius=1e4, min=1e-32, max=1e16)
    # strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5**4)
    strategy = pp.optim.strategy.Adaptive()

    optimizer = LM(graph, solver=solver, strategy=strategy, min=1e-10, reject=30)
    scheduler = StopOnPlateau(optimizer, steps=20, patience=3, decreasing=1e-7, verbose=True)

    pngname = os.path.join(args.save, args.dataname+'.png')
    axlim = plot_and_save(pp.SE3(graph.nodes).translation(), pngname, args.dataname)
    axlim = None
    ### the 1st implementation: for customization and easy to extend
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(10):
        loss = optimizer.step(input=input, weight=infos)
        scheduler.step(loss)

        name = os.path.join(args.save, args.dataname + '_' + str(scheduler.steps))
        title = 'PGO at the %d step(s) with loss %7f'%(scheduler.steps, loss.item())
    torch.cuda.synchronize()
    end.record()
    print('Time elapsed: %.3f ms'%(start.elapsed_time(end)))
    print('Final loss: %7f'%(loss.item()/2))
    plot_and_save(pp.SE3(graph.nodes).translation(), name+'.png', title, axlim=axlim)
    torch.save(graph.state_dict(), name+'.pt')
    write_ceres_txt(graph.nodes, name+'.txt')

    ### The 2nd implementation: equivalent to the 1st one, but more compact
    # scheduler.optimize(input=(edges, poses, infos))
