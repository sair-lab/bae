from functools import partial
import torch
from bae.utils.pysolvers import cuSolverSP as cudss
import scipy.sparse.linalg as spla
import scipy.sparse as sp


if __name__ == '__main__':
    spd = torch.rand(4, 3)
    A = spd.T @ spd
    print(A)
    b = torch.rand(3).to(torch.float64).cuda()
    print(b)
    A = A.to_sparse_csr().to(torch.float64).cuda()
    A = torch.sparse_csr_tensor(torch.tensor(A.crow_indices(), dtype=torch.int32),
                                torch.tensor(A.col_indices(), dtype=torch.int32), torch.tensor(A.values()),
                                dtype=torch.double)
    print(A)
    x = cudss(A, b)
    # print(x)
    print((A @ x - b).norm())
    A_csr = sp.csr_matrix((A.values().cpu().numpy(),
                                A.col_indices().cpu().numpy(),
                                A.crow_indices().cpu().numpy()),
                                shape=A.shape)
    b = b.cpu().numpy()
    x = spla.spsolve(A_csr, b, use_umfpack=False)
    # print(x)
    print(A_csr @ x - b)