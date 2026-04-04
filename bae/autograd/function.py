import torch
import numpy as np
import pypose as pp

WHITELISTED_MAPS = tuple(
    func for func in (
        torch._C.TensorBase.__add__,
        torch._C.TensorBase.__sub__,
        torch._C.TensorBase.__mul__,
        getattr(torch._C.TensorBase, "__div__", None),
        torch._C.TensorBase.add,
        torch._C.TensorBase.sub,
        torch._C.TensorBase.mul,
    ) if func is not None
)

_LTYPE_PRESERVING_FUNCS = {
    torch._C.TensorBase.__getitem__,
    torch.cat,
    *WHITELISTED_MAPS,
    torch._C.TensorBase.clone,
    torch._C.TensorBase.to,
}


def _iter_tracked_tensors(values):
    if isinstance(values, torch.Tensor):
        yield values
    elif isinstance(values, (tuple, list)):
        for value in values:
            yield from _iter_tracked_tensors(value)


def _merge_optrace(values):
    merged_optrace = {}
    for value in _iter_tracked_tensors(values):
        if hasattr(value, 'optrace'):
            merged_optrace.update(value.optrace)
    return merged_optrace


def _attach_index_trace(result, index, tensor):
    if not hasattr(result, 'optrace'):
        result.optrace = {}
    result.optrace[id(result)] = ("index", index, tensor)
    return result


def _attach_cat_trace(result, tensors, dim):
    merged_optrace = _merge_optrace(tensors)
    merged_optrace[id(result)] = ("cat", dim, tuple(tensors))
    result.optrace = merged_optrace
    return result


def _attach_map_trace(result, func, args):
    merged_optrace = _merge_optrace(args)
    merged_optrace[id(result)] = ("map", func, args)
    result.optrace = merged_optrace
    return result


def _find_tracking_source(values, cls):
    for value in _iter_tracked_tensors(values):
        if isinstance(value, cls):
            return value
    return None


def _retain_ltype(result, tracking_source, cls, func):
    if tracking_source is None or not issubclass(cls, pp.LieTensor):
        return result
    if func not in _LTYPE_PRESERVING_FUNCS:
        return result
    if not isinstance(result, torch.Tensor) or isinstance(result, cls):
        return result
    if result.shape[-1:] != tracking_source.ltype.dimension:
        return result
    wrapped = torch.Tensor.as_subclass(result, cls)
    wrapped.ltype = tracking_source.ltype
    return wrapped

# =============================================================================
# Class: IndexTrackingTensor
# A custom subclass of torch.Tensor that tracks the indices used for slicing.
# When an instance is sliced via __getitem__, it records the provided index.
# =============================================================================
class TrackingTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, *args, **kwargs):
        if cls is TrackingTensor and isinstance(data, pp.LieTensor):
            return _TrackingLieTensor(data, *args, **kwargs)

        if isinstance(data, torch.Tensor):
            return torch.Tensor._make_subclass(cls, data, *args, **kwargs)
        return torch.Tensor._make_subclass(cls, torch.as_tensor(data), *args, **kwargs)


    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        result = super(TrackingTensor, cls).__torch_function__(func, types, args=args, kwargs=kwargs)
        result = _retain_ltype(result, _find_tracking_source(args, cls), cls, func)

        if isinstance(result, torch.Tensor) and (not args or getattr(args[0], '_active', True)):
            if (func == torch._C.TensorBase.__getitem__) and isinstance(args[1], torch.Tensor):
                result = _attach_index_trace(result, args[1], args[0])
            elif func == torch.cat:
                tensors = args[0]
                dim = kwargs.get("dim", 0)
                if len(args) > 1:
                    dim = args[1]
                if dim != 0:
                    raise NotImplementedError("TrackingTensor only supports torch.cat(..., dim=0)")
                result = _attach_cat_trace(result, tensors, dim)
            elif func in WHITELISTED_MAPS:
                result = _attach_map_trace(result, func, args)
        return result
    
    def __getitem__(self, index):
        # if isinstance(index, (slice, list, np.ndarray)):
        #     index = self._convert_to_index_tensor(index)
        # elif isinstance(index, tuple):
        #     if index[0] == slice(None, None, None):
        #         ...
        #         # this belongs to the mapping type
        #     else:
        #         index = (self._convert_to_index_tensor(index[0]), *index[1:])
        result = super().__getitem__(index)
        # if getattr(self, '_active', True):
        #     print(f"__getitem__ called with index {index}")
        #     if isinstance(index, torch.Tensor):
        #         index_edge = ("index", index, self)
        #         result.optrace[id(result)] = index_edge
        return result

    def _convert_to_index_tensor(self, index):
        if isinstance(index, int):
            return index
        elif isinstance(index, slice):
            start = 0 if index.start is None else index.start
            stop = index.stop  # assume stop is provided
            step = 1 if index.step is None else index.step
            return torch.arange(start, stop, step)
        elif isinstance(index, list):
            return torch.tensor(index)
        elif isinstance(index, np.ndarray):
            return torch.from_numpy(index)
        
    def tensor(self) -> torch.Tensor:
        return torch.Tensor.as_subclass(self, torch.Tensor)


class _TrackingLieTensor(TrackingTensor, pp.LieTensor):
    def __init__(self, data=None, *args, **kwargs):
        if isinstance(data, pp.LieTensor):
            self.ltype = data.ltype

    @staticmethod
    def __new__(cls, data, *args, **kwargs):
        if not isinstance(data, pp.LieTensor):
            raise TypeError(f"_TrackingLieTensor expects a LieTensor input, got {type(data)!r}")
        instance = torch.Tensor.as_subclass(data, cls)
        instance.ltype = data.ltype
        return instance

    def detach(self):
        detached = torch.Tensor.as_subclass(super().detach(), type(self))
        detached.ltype = self.ltype
        return detached
"""
graph design
Node: (tensor_type: [nn.Parameter, tensor, pp.LieTensor])
Edge: (indexing, mapping)

G: (V: [Node...], E: [Edge...])

for each e = (u, v) \in E
parent[loss] = (project, [camera_indexed, point_indexed])
parent[camera_indexed] = ((indexing, indices), camera_parameters)

build
parent: key: id(tensor), value: map edge (edge_type, func, [input_args]), index edge (edge_type, indicies, orig_arg)

backward
1. access loss.parent
2. check edge type
3.1. if indexing, permute value
3.2. if mapping, revise value

recusively call 1-3 until input node is reached. 
"""


# =============================================================================
# Decorator: index_transform
# A decorator that wraps a function to transform indices.
# It prints a message, calls the function, and attaches metadata recording
# the tracked indices from any IndexTrackingTensor arguments.
# =============================================================================
def index_transform(tensor, index):
    result = tensor[index]
    return _attach_index_trace(result, index, tensor)


# =============================================================================
# Decorator: map_transform
# A decorator that wraps a function to apply a map transformation.
# It unsqueezes tensor arguments, calls the function, then squeezes the result.
# Additionally, it merges metadata from all input tensors.
# =============================================================================
def map_transform(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        # map edge (edge_type, func, [input_args])
        # ensure final result is an IndexTrackingTensor
        return _attach_map_trace(result, func, args)
    return wrapper

    # map_transform(vmap(func))
    # this is wrong: vmap(map_transform(func))
