import torch
from bae.sparse.py_ops import jtj_diag

crow_indices = torch.tensor([0, 2, 4])
col_indices = torch.tensor([0, 1, 0, 1])
values = torch.tensor([[[0, 1, 2], [6, 7, 8]],
                        [[3, 4, 5], [9, 10, 11]],
                        [[12, 13, 14], [18, 19, 20]],
                        [[15, 16, 17], [21, 22, 23]]])
bsr = torch.sparse_bsr_tensor(crow_indices, col_indices, values, dtype=torch.float64)
output = jtj_diag(bsr)

coo = bsr.to_sparse_coo()
assert torch.isclose((coo @ coo.mT).to_dense().diag(), output).all()