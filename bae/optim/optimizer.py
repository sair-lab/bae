from functools import partial
import torch
from pypose.optim import LevenbergMarquardt as ppLM
import pypose as pp
from ..autograd.graph import jacobian
from ..autograd.function import TrackingTensor
from ..sparse.py_ops import diagonal_op_
from ..sparse.spgemm import CuSparse
from ..utils.parameter import parameter_update_shape




class LM(ppLM):
    def __init__(self, *args, **kwargs):
        super(LM, self).__init__(*args, **kwargs)
        self.mm = CuSparse()

    @torch.no_grad()
    def step(self, input, target=None, weight=None):
        for pg in self.param_groups:
            weight = self.weight if weight is None else weight
            R = list(self.model(input))
            R = R[0]
            J = jacobian(R, pg['params'])
            if isinstance(R, TrackingTensor):
                R = R.tensor()
            J = torch.cat([j.to_sparse_coo() for j in J], dim=-1)

            self.last = self.loss = self.loss if hasattr(self, 'loss') else self.model.loss(input, target)
            J_T = J.mT
            self.reject_count = 0
            J_T = J_T.to_sparse_csr()
            J = J.to_sparse_csr()
            A = self.mm(J_T, J)

            diagonal_op_(A, op=partial(torch.clamp_, min=pg['min'], max=pg['max']))

            while self.last <= self.loss:
                diagonal_op_(A, op=partial(torch.mul, other=1+pg['damping']))
                try:
                    D = self.solver(A, -J_T @ R.view(-1, 1))
                except Exception as e:
                    print(e, "\nLinear solver failed. Breaking optimization step...")
                    break
                self.update_parameter(pg['params'], D)
                self.loss = self.model.loss(input, target)
                print("Loss:", self.loss, "Last Loss:", self.last, "Reject Count:", self.reject_count, "Damping:", pg['damping'])
                self.strategy.update(pg, last=self.last, loss=self.loss, J=J, D=D, R=R.view(-1, 1))
                if self.last < self.loss and self.reject_count < self.reject:  # reject step
                    self.update_parameter(params=pg['params'], step=-D)
                    self.loss, self.reject_count = self.last, self.reject_count + 1
                else:
                    break
        return self.loss

    def update_parameter(self, params, step):
        numels = []
        for param in params:
            if param.requires_grad:
                numels.append(torch.Size(parameter_update_shape(param)).numel())
        steps = step.split(numels)
        for (param, d) in zip(params, steps):
            if param.requires_grad:
                step_view = d.view(parameter_update_shape(param))
                if getattr(param, 'trim_SE3_grad', False):
                    param[..., :7] = pp.SE3(param[..., :7]).add_(pp.se3(step_view[..., :6]))
                    if param.shape[-1] > 7:
                        param[:, 7:] += step_view[..., 6:]
                else:
                    param.add_(step_view)
