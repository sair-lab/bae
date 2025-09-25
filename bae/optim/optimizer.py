from functools import partial
import math
import torch
from pypose.optim import LevenbergMarquardt as ppLM
import pypose as pp
from ..autograd.graph import backward, construct_sbt
from ..autograd.function import TrackingTensor
from ..sparse.py_ops import diagonal_op_
from ..sparse.spgemm import CuSparse

def jacobian(output, params):
    assert output.optrace[id(output)][0] == 'map', "The last operation in compute graph being indexing transform is not meaningful"
    backward(output)
    res = []
    for param in params:
        if hasattr(param, 'jactrace'):
            if getattr(param, 'trim_SE3_grad', False):
                if isinstance(param.jactrace, tuple):
                    values = param.jactrace[1]
                elif isinstance(param.jactrace, torch.Tensor) and param.jactrace.layout == torch.sparse_bsr:
                    values = param.jactrace.values()
                else:
                    values = param.jactrace

                if values.shape[-1] == 7:
                    values = values[..., :6]
                else:
                    values = torch.cat([values[..., :6], values[..., 7:]], dim=-1)
                
                if isinstance(param.jactrace, tuple):
                    param.jactrace = (param.jactrace[0], values)
                elif isinstance(param.jactrace, torch.Tensor) and param.jactrace.layout == torch.sparse_bsr:
                    param.jactrace = torch.sparse_bsr_tensor(
                        col_indices=param.jactrace.col_indices(), 
                        crow_indices=param.jactrace.crow_indices(),
                        values=values,
                        size=(param.jactrace.shape[0], param.shape[0] * values.shape[-1]),
                        device=param.device,
                    )
                else:
                    param.jactrace = values
            if type(param.jactrace) is tuple:
                param.jactrace = construct_sbt(param.jactrace[1], param.shape[0], param.jactrace[0], type=torch.sparse_bsr)
            res.append(param.jactrace)
            delattr(param, 'jactrace')
            
    return res



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
                    D = D[:, None]
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
                if getattr(param, 'trim_SE3_grad', False):
                    numels.append(math.prod(param.shape[:-1]) * (param.shape[-1] - 1))
                else:
                    numels.append(param.numel())
        steps = step.split(numels)
        for (param, d) in zip(params, steps):
            if param.requires_grad:
                if getattr(param, 'trim_SE3_grad', False):
                    param[..., :7] = pp.SE3(param[..., :7]).add_(pp.se3(d.view(param.shape[0], -1)[..., :6]))
                    if param.shape[-1] > 7:
                        param[:, 7:] += d.view(param.shape[0], -1)[:, 6:]
                else:
                    param.add_(d.view(param.shape))
