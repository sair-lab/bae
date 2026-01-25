
from typing import Optional

import torch
from torch.func import jacrev


def construct_sbt(jac_from_vmap, num, index: Optional[torch.Tensor], type=torch.sparse_bsc):
    if index is None:
        index = torch.arange(num, device=jac_from_vmap.device, dtype=torch.int32)
    n = index.shape[0] # num 2D points
    block_shape = jac_from_vmap.shape[1:]
    idx_dtype = index.dtype

    if type == torch.sparse_bsc:
        i = torch.stack([torch.arange(n, dtype=index.dtype, device=index.device), index])
        dummy_val = torch.arange(n, device=index.device, dtype=torch.int32)
        dummy_coo = torch.sparse_coo_tensor(i, dummy_val, size=(n, num), device=index.device, dtype=torch.int32)
        dummy_csc = dummy_coo.coalesce().to_sparse_csc()
        return torch.sparse_bsc_tensor(ccol_indices=dummy_csc.ccol_indices().to(torch.int32), 
                                    row_indices=dummy_csc.row_indices().to(torch.int32),
                                    values = jac_from_vmap[dummy_csc.values()],
                                    size = (n * block_shape[0], num * block_shape[1]),
                                    device=index.device, dtype=jac_from_vmap.dtype)
    elif type == torch.sparse_bsr:
        return torch.sparse_bsr_tensor(col_indices=index, 
                                    crow_indices=torch.arange(n + 1, device=index.device, dtype=idx_dtype),
                                    values = jac_from_vmap,
                                    size = (n * block_shape[0], num * block_shape[1]),
                                    device=index.device, dtype=jac_from_vmap.dtype)

def amend_trace(arg, jac_trace: tuple):
    if hasattr(arg, 'jactrace'):  # convert to sparse_bsr needed for accumulation
        if type(arg.jactrace) is tuple and type(jac_trace) is tuple:
            if arg.jactrace[0] is None and jac_trace[0] is None:
                arg.jactrace = (None, arg.jactrace[1] + jac_trace[1])
                return 
        if type(arg.jactrace) is tuple:
            arg.jactrace = construct_sbt(arg.jactrace[1], arg.shape[0], arg.jactrace[0], type=torch.sparse_bsr)
        if type(jac_trace) is tuple:
            jac_trace = construct_sbt(jac_trace[1], arg.shape[0], jac_trace[0], type=torch.sparse_bsr)
        arg.jactrace = arg.jactrace + jac_trace
    else:
        arg.jactrace = jac_trace

def update_from_trace(bsrt: torch.Tensor, arg, new_col: Optional[torch.Tensor]=None, new_val: Optional[torch.Tensor]=None):
    if new_col is not None:
        jac_trace = torch.sparse_bsr_tensor(
                col_indices=new_col, 
                crow_indices=bsrt.crow_indices(),
                values=bsrt.values(),
                size=(bsrt.shape[0], arg.shape[0] * bsrt.values().shape[-1]),
                device=bsrt.device,
            )
    if new_val is not None:
        jac_trace = torch.sparse_bsr_tensor(
                col_indices=bsrt.col_indices(), 
                crow_indices=bsrt.crow_indices(),
                values=new_val,
                size=(bsrt.shape[0], arg.shape[0] * new_val.shape[-1]),
                device=bsrt.device,
            )
    return jac_trace

def backward(output_):
    if output_.optrace[id(output_)][0] == 'map':
        func = output_.optrace[id(output_)][1]
        args = output_.optrace[id(output_)][2]
        argnums = tuple(idx for idx, arg in enumerate(args) if hasattr(arg, 'optrace') or isinstance(arg, torch.nn.Parameter))
        if len(argnums) == 0:
            warning("No upstream parameters to compute jacobian")
            return
        jac_blocks = torch.vmap(jacrev(func, argnums=argnums))(*args)
        for jacidx, argidx in enumerate(argnums):
            jac_block = jac_blocks[jacidx]
            arg = args[argidx]
            assert jac_block.ndim == 3, "`func` is not properly vectorized in `torch.vmap`"
            # TODO: perhaps flatten the jacobian block in the future
            if not hasattr(output_, 'jactrace'):  # check for upstream jacobian
                jac_trace = (None, jac_block)  # leave None for identity indices
            else:
                indices = None
                if type(output_.jactrace) is tuple:
                    indices = output_.jactrace[0]
                    jac_ustrm = output_.jactrace[1]
                elif type(output_.jactrace) is torch.Tensor and output_[jacidx].jactrace.layout == torch.sparse_bsr:
                    indices = output_.jactrace.col_indices()
                    jac_ustrm = output_.jactrace.values()

                if indices is not None:
                    jac_block = jac_block[indices]
                jac_block = jac_ustrm @ jac_block
                
                if type(output_.jactrace) is tuple:
                    jac_trace = (indices, jac_block)
                elif type(output_.jactrace) is torch.Tensor and output_.jactrace.layout == torch.sparse_bsr:
                    jac_trace = update_from_trace(output_.jactrace, arg, new_val=jac_block)
            amend_trace(arg, jac_trace)
        for argidx in argnums:
            if isinstance(args[argidx], torch.Tensor) and hasattr(args[argidx], 'optrace'):
                backward(args[argidx])


    elif output_.optrace[id(output_)][0] == 'index':
        index = output_.optrace[id(output_)][1]
        arg = output_.optrace[id(output_)][2]

        # If the last operation is indexing, there is no downstream map op to
        # populate Jacobian values. In this case, the Jacobian block values are
        # identity matrices placed at the indexed columns.
        if not hasattr(output_, 'jactrace'):
            if output_.ndim == 1:
                eye_blocks = torch.ones((output_.shape[0], 1, 1), device=output_.device, dtype=output_.dtype)
            else:
                block_dim = output_.shape[-1]
                eye = torch.eye(block_dim, device=output_.device, dtype=output_.dtype)
                eye_blocks = eye.unsqueeze(0).repeat(output_.shape[0], 1, 1)
            output_.jactrace = (None, eye_blocks)

        if type(output_.jactrace) is tuple:
            if output_.jactrace[0] is not None:
                upstream_index = output_.jactrace[0]
                index = index[upstream_index]
                jac_trace = (index, output_.jactrace[1])
            elif output_.jactrace[0] is None:
                jac_trace = (index, output_.jactrace[1])
        elif type(output_.jactrace) is torch.Tensor and output_.jactrace.layout == torch.sparse_bsr:
            upstream_index = output_.jactrace.col_indices()
            index = index[upstream_index]
            jac_trace = update_from_trace(output_.jactrace, arg, new_col=index)
            
        amend_trace(arg, jac_trace)
        if isinstance(arg, torch.Tensor) and hasattr(arg, 'optrace'):
            backward(arg)


def jacobian(output, params):
    assert output.optrace[id(output)][0] in ('map', 'index'), "Unsupported last operation in compute graph"
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
