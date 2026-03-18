import os
import time
import torch
import argparse
import pypose as pp
from torch import nn
from bae.autograd.function import TrackingTensor, map_transform
from bae.utils.ceres_pose import (
    quat_inv_xyzw,
    quat_mul_xyzw,
    quat_rotate_xyzw,
)

from bae.utils.pgo_dataset import G2OPGO
from bae.utils.pgo import plot_and_save
from pypose.optim.scheduler import StopOnPlateau
from bae.utils.pysolvers import PCG
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

def _pose_graph_residual(poses, node1, node2, infos):
    if isinstance(poses, TrackingTensor):
        poses = poses.tensor()
    if isinstance(node1, TrackingTensor):
        node1 = node1.tensor()
    if isinstance(node2, TrackingTensor):
        node2 = node2.tensor()
    if isinstance(infos, TrackingTensor):
        infos = infos.tensor()

    p_a, q_a = node1[..., :3], node1[..., 3:7]
    p_b, q_b = node2[..., :3], node2[..., 3:7]
    p_meas, q_meas = poses[..., :3], poses[..., 3:7]

    q_a_inv = quat_inv_xyzw(q_a)
    q_ab_est = quat_mul_xyzw(q_a_inv, q_b)
    p_ab_est = quat_rotate_xyzw(q_a_inv, p_b - p_a)

    r_p = p_ab_est - p_meas

    # Match Ceres pose_graph_3d: 2 * vec(q_meas * q_est^{-1}).
    delta_q = quat_mul_xyzw(q_meas, quat_inv_xyzw(q_ab_est))
    r_q = 2.0 * delta_q[..., :3]

    residual = torch.cat((r_p, r_q), dim=-1)
    residual = infos @ residual[..., None]
    return residual[..., 0]


@map_transform
def _tracked_pose_graph_residual(poses, node1, node2, infos):
    return _pose_graph_residual(poses, node1, node2, infos)

class PoseGraph(nn.Module):

    def __init__(self, nodes):
        super().__init__()
        self.nodes = nn.Parameter(TrackingTensor(nodes))
        self.nodes.trim_SE3_grad = True
        self.nodes.ceres_pose_grad = True

    def forward(self, edges, poses, infos):
        node1 = self.nodes[edges[..., 0]]
        node2 = self.nodes[edges[..., 1]]
        return _tracked_pose_graph_residual(poses, node1, node2, infos)


class PoseGraphFixedFirst(nn.Module):
    def __init__(self, nodes_rest):
        super().__init__()
        self.nodes_rest = nn.Parameter(TrackingTensor(nodes_rest))
        self.nodes_rest.trim_SE3_grad = True
        self.nodes_rest.ceres_pose_grad = True

    def nodes_all(self, node_fixed):
        return torch.cat([node_fixed, self.nodes_rest], dim=0)

    def forward(self, edges, poses, infos, node_fixed):
        nodes = self.nodes_all(node_fixed)
        node1 = nodes[edges[..., 0]]
        node2 = nodes[edges[..., 1]]
        return _tracked_pose_graph_residual(poses, node1, node2, infos)


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
    parser.add_argument("--steps", type=int, default=80, \
                        help="number of LM outer iterations")
    parser.add_argument('--no-gauge-fix', dest='gauge_fix', action='store_false', \
                        help='optimize all nodes (disables fixed-first-node gauge constraint)')
    parser.set_defaults(gauge_fix=True)
    parser.set_defaults(vectorize=True)
    args = parser.parse_args(); print(args)
    os.makedirs(os.path.join(args.save), exist_ok=True)

    data = G2OPGO(args.dataroot, args.dataname, device=args.device, download=True)
    if isinstance(data.nodes, pp.LieTensor):
        data.nodes = data.nodes.tensor()
    if isinstance(data.poses, pp.LieTensor):
        data.poses = data.poses.tensor()
    data.nodes = data.nodes.to(DTYPE)
    data.poses = data.poses.to(DTYPE)
    data.infos = data.infos.to(DTYPE)

    edges, poses, infos = data.edges, data.poses, data.infos
    infos = torch.linalg.cholesky(infos)
    if args.gauge_fix:
        if data.nodes.shape[0] < 2:
            raise ValueError("Gauge-fix mode requires at least two nodes.")
        node_fixed = data.nodes[:1].clone()
        input = {'edges': edges, 'poses': poses, 'infos': infos, 'node_fixed': node_fixed}
        graph = PoseGraphFixedFirst(data.nodes[1:]).to(args.device)
    else:
        input = {'edges': edges, 'poses': poses, 'infos': infos}
        graph = PoseGraph(data.nodes).to(args.device)
    solver = PCG(tol=1e-5)
    # solver = ppos.Cholesky()
    # strategy = ppost.TrustRegion(radius=1e4, min=1e-32, max=1e16)
    # strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5**4)
    strategy = pp.optim.strategy.Adaptive()

    optimizer = LM(graph, solver=solver, strategy=strategy, min=1e-10, reject=30)
    scheduler = StopOnPlateau(optimizer, steps=20, patience=3, decreasing=1e-7, verbose=True)

    if args.gauge_fix:
        nodes_current = graph.nodes_all(input['node_fixed'])
    else:
        nodes_current = graph.nodes

    pngname = os.path.join(args.save, args.dataname+'.png')
    axlim = plot_and_save(pp.SE3(nodes_current).translation(), pngname, args.dataname)
    axlim = None
    ### the 1st implementation: for customization and easy to extend
    if args.device == 'cuda':
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        start = time.perf_counter()
    for i in range(args.steps):
        loss = optimizer.step(input=input, weight=infos)
        scheduler.step(loss)

        name = os.path.join(args.save, args.dataname + '_' + str(scheduler.steps))
        title = 'PGO at the %d step(s) with loss %7f'%(scheduler.steps, loss.item())
    if args.device == 'cuda':
        torch.cuda.synchronize()
        end.record()
        print('Time elapsed: %.3f ms'%(start.elapsed_time(end)))
    else:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        print('Time elapsed: %.3f ms'%(elapsed_ms))
    print('Final loss: %7f'%(loss.item()/2))
    if args.gauge_fix:
        nodes_current = graph.nodes_all(input['node_fixed'])
    else:
        nodes_current = graph.nodes
    plot_and_save(pp.SE3(nodes_current).translation(), name+'.png', title, axlim=axlim)
    torch.save(graph.state_dict(), name+'.pt')
    write_ceres_txt(nodes_current, name+'.txt')

    ### The 2nd implementation: equivalent to the 1st one, but more compact
    # scheduler.optimize(input=(edges, poses, infos))
