
# https://nvidia.github.io/warp/modules/sparse.html
import warp as wp
from warp import sparse as wps
from warp.optim import linear as wpol
import torch
wp.init()


def torchbsr2wp(tbsr):
    assert tbsr.layout == torch.sparse_bsr
    block_type = wp.mat(shape=tbsr.values().shape[-2:], dtype=wp.dtype_from_torch(tbsr.dtype))
    bsr = wps.bsr_matrix_t(block_type)()
    bsr.nrow = int(tbsr.shape[0] // block_type._shape_[0])
    bsr.ncol = int(tbsr.shape[1] // block_type._shape_[1])
    bsr.nnz = int(tbsr.values().shape[0])
    bsr.columns = wp.from_torch(tbsr.col_indices().to(torch.int32))
    bsr.values = wp.from_torch(tbsr.values().contiguous(), dtype=block_type)
    bsr.offsets = wp.from_torch(tbsr.crow_indices().to(torch.int32))
    return bsr

def wp2torchbsr(bsr):
    """
    Convert a Warp BSR matrix back into a PyTorch sparse BSR tensor.

    Args:
        bsr (warp.bsr_matrix_t): A Warp BSR matrix.

    Returns:
        torch.Tensor: A PyTorch sparse BSR tensor with the same data.
    """

    # Extract block dimensions from the values array:
    # Typically: bsr.values has shape (nnz, block_rows, block_cols)
    # so we take the last two dimensions as the block shape.
    block_rows, block_cols = bsr.block_shape[-2:]

    # Total matrix dimensions:
    nrows = bsr.nrow * block_rows
    ncols = bsr.ncol * block_cols

    # Convert Warp arrays back to PyTorch tensors
    crow = wp.to_torch(bsr.offsets)            # shape [bsr.nrow + 1]
    col  = wp.to_torch(bsr.columns)            # shape [bsr.nnz]
    vals = wp.to_torch(bsr.values)             # shape [bsr.nnz, block_rows, block_cols]

    tbsr = torch.sparse_bsr_tensor(
        crow,
        col,
        vals,
        size=(nrows, ncols),
        dtype=vals.dtype,
        device=vals.device
    )

    return tbsr

def format_vec_for_bsr(tvec, block_shape):
    y_vec_len = block_shape[1]
    y_dtype = wp.vec(length=y_vec_len, dtype=wp.dtype_from_torch(tvec.dtype))
    if tvec.ndim == 1 and tvec.shape[-1] != y_vec_len:
        tvec = tvec.reshape(-1, y_vec_len)
    vwp = wp.from_torch(tvec, dtype=y_dtype)
    return vwp

def _sparse_csr_add(input, other, alpha=1.0):
    input = torchbsr2wp(input)
    other = torchbsr2wp(other)
    res = wps.bsr_axpy(other, input, alpha=alpha)
    res = wp2torchbsr(res)
    return res

from torch.library import Library
sparse_lib = Library('aten', 'IMPL')
sparse_lib.impl('add.Tensor', _sparse_csr_add, 'SparseCsrCUDA')
# this will, however, invalidate add_sparse_csr


if __name__ == '__main__':
    crow_indices = torch.tensor([0, 2, 4], dtype=torch.int32)
    col_indices = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
    values = torch.tensor([[[0, 1, 2], [6, 7, 8]],
                        [[3, 4, 5], [9, 10, 11]],
                        [[12, 13, 14], [18, 19, 20]],
                        [[15, 16, 17], [21, 22, 23]]])

    bsr = torch.sparse_bsr_tensor(crow_indices, col_indices, values, dtype=torch.float32).to('cuda')
    print(bsr.to_dense())
    a = torchbsr2wp(bsr)
    print(a.transpose() @ a)
    print(wp2torchbsr(a).to_dense())
    # print(wpol.preconditioner(a))