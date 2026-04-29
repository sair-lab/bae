import os
os.environ.setdefault('BAE_USE_PYPOSE_AMBIENT_GRAD', '1')

import time
import torch
import argparse
import pypose as pp
from torch import nn
from bae.autograd.function import TrackingTensor, map_transform

from bae.utils.pgo_dataset import G2OPGO
from bae.utils.pgo import plot_and_save, render_frame, save_gif
from pypose.optim.scheduler import StopOnPlateau
from bae.utils.pysolvers import PCG
from bae.optim import LM

OPTIMIZE_INTRINSICS = False
USE_QUATERNIONS=True
DTYPE_CHOICES = {
    'float64': torch.float64,
    'fp64': torch.float64,
    'float32': torch.float32,
    'fp32': torch.float32,
}

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
    if isinstance(infos, TrackingTensor):
        infos = infos.tensor()

    pose_ab_est = node1.Inv() @ node2
    r_p = pose_ab_est.translation() - poses.translation()
    # Match Ceres pose_graph_3d: 2 * vec(q_meas * q_est^{-1}).
    delta_q = poses.rotation() @ pose_ab_est.rotation().Inv()
    r_q = 2.0 * delta_q.tensor()[..., :3]

    residual = torch.cat((r_p, r_q), dim=-1)
    residual = infos @ residual[..., None]
    return residual[..., 0]


@map_transform
def pose_graph_residual(poses, node1, node2, infos):
    return _pose_graph_residual(poses, node1, node2, infos)

class PoseGraph(nn.Module):

    def __init__(self, nodes):
        super().__init__()
        self.nodes = nn.Parameter(TrackingTensor(nodes))

    def forward(self, edges, poses, infos):
        node1 = self.nodes[edges[..., 0]]
        node2 = self.nodes[edges[..., 1]]
        return pose_graph_residual(poses, node1, node2, infos)


class PoseGraphFixedFirst(nn.Module):
    def __init__(self, nodes_rest):
        super().__init__()
        self.nodes_rest = nn.Parameter(TrackingTensor(nodes_rest))

    def nodes_all(self, node_fixed):
        return torch.cat([node_fixed, self.nodes_rest], dim=0)

    def forward(self, edges, poses, infos, node_fixed):
        nodes = self.nodes_all(node_fixed)
        node1 = nodes[edges[..., 0]]
        node2 = nodes[edges[..., 1]]
        return pose_graph_residual(poses, node1, node2, infos)


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
    parser.add_argument('--gif', action='store_true', \
                        help='save an animated GIF of optimization progress')
    parser.add_argument('--gif-every', type=int, default=1, \
                        help='capture every N optimization steps in the GIF')
    parser.add_argument('--gif-duration', type=int, default=250, \
                        help='GIF frame duration in milliseconds')
    parser.add_argument('--no-vectorize', dest='vectorize', action='store_false', \
                        help="to save memory")
    parser.add_argument('--vectorize', action='store_true', \
                        help='to accelerate computation')
    parser.add_argument("--steps", type=int, default=80, \
                        help="number of LM outer iterations")
    parser.add_argument('--dtype', type=str, default='float64', choices=tuple(DTYPE_CHOICES.keys()), \
                        help='parameter / residual dtype')
    parser.add_argument('--no-gauge-fix', dest='gauge_fix', action='store_false', \
                        help='optimize all nodes (disables fixed-first-node gauge constraint)')
    parser.set_defaults(gauge_fix=True)
    parser.set_defaults(vectorize=True)
    args = parser.parse_args(); print(args)
    os.makedirs(os.path.join(args.save), exist_ok=True)
    if args.gif_every < 1:
        raise ValueError('--gif-every must be >= 1')
    dtype = DTYPE_CHOICES[args.dtype]

    data = G2OPGO(args.dataroot, args.dataname, device=args.device, download=True)
    data.nodes = data.nodes.to(dtype)
    data.poses = data.poses.to(dtype)
    data.infos = data.infos.to(dtype)

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
    strategy = pp.optim.strategy.Adaptive()

    optimizer = LM(graph, solver=solver, strategy=strategy, min=1e-10, reject=30)
    scheduler = StopOnPlateau(optimizer, steps=20, patience=3, decreasing=1e-7, verbose=True)

    if args.gauge_fix:
        nodes_current = graph.nodes_all(input['node_fixed'])
    else:
        nodes_current = graph.nodes

    sample_prefix = os.path.join(args.save, os.path.splitext(args.dataname)[0])
    plot_and_save(nodes_current.translation(), sample_prefix + '.png', args.dataname)

    gif_frames = []
    if args.gif:
        frame, _ = render_frame(nodes_current.translation(), args.dataname)
        gif_frames.append(frame)

    if args.device == 'cuda':
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        start = time.perf_counter()
    for i in range(args.steps):
        loss = optimizer.step(input=input, weight=infos)
        scheduler.step(loss)

        name = os.path.join(args.save, os.path.splitext(args.dataname)[0] + '_' + str(scheduler.steps))
        title = 'PGO at the %d step(s) with loss %7f'%(scheduler.steps, loss.item())
        if args.gif and ((i + 1) % args.gif_every == 0 or i == args.steps - 1):
            if args.gauge_fix:
                nodes_current = graph.nodes_all(input['node_fixed'])
            else:
                nodes_current = graph.nodes
            frame, _ = render_frame(nodes_current.translation(), title)
            gif_frames.append(frame)
    if args.device == 'cuda':
        end.record()
        torch.cuda.synchronize()
        print('Time elapsed: %.3f ms'%(start.elapsed_time(end)))
    else:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        print('Time elapsed: %.3f ms'%(elapsed_ms))
    print('Final loss: %7f'%(loss.item()/2))
    if args.gauge_fix:
        nodes_current = graph.nodes_all(input['node_fixed'])
    else:
        nodes_current = graph.nodes
    plot_and_save(nodes_current.translation(), name+'.png', title)
    torch.save(graph.state_dict(), name+'.pt')
    write_ceres_txt(nodes_current.tensor(), name+'.txt')
    if args.gif:
        save_gif(gif_frames, sample_prefix + '.gif', duration=args.gif_duration)

    ### The 2nd implementation: equivalent to the 1st one, but more compact
    # scheduler.optimize(input=(edges, poses, infos))
