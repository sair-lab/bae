import torch
import numpy as np
import pypose as pp

WHITELISTED_MAPS = (torch._C.TensorBase.__add__, 
                    torch._C.TensorBase.__sub__, 
                    torch._C.TensorBase.__mul__, 
                    torch._C.TensorBase.__div__,
                    torch._C.TensorBase.add,
                    torch._C.TensorBase.sub,
                    torch._C.TensorBase.mul,)

# =============================================================================
# Class: IndexTrackingTensor
# A custom subclass of torch.Tensor that tracks the indices used for slicing.
# When an instance is sliced via __getitem__, it records the provided index.
# =============================================================================
class TrackingTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, *args, **kwargs):
        if isinstance(data, torch.Tensor):
            instance = torch.Tensor._make_subclass(cls, data, *args, **kwargs)
        else:
            instance = torch.Tensor._make_subclass(cls, torch.as_tensor(data), *args, **kwargs)
        return instance


    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        result = super(TrackingTensor, cls).__torch_function__(func, types, args=args, kwargs=kwargs)
        
        if isinstance(result, torch.Tensor) and getattr(args[0], '_active', True):
            # print(f"__torch_function__ called with {func}")
            if (func == torch._C.TensorBase.__getitem__) and isinstance(args[1], torch.Tensor):
                if not hasattr(result, 'optrace'):
                    result.optrace = {}
                index_edge = ("index", args[1], args[0])
                result.optrace[id(result)] = index_edge
            elif func in WHITELISTED_MAPS:
                merged_optrace = {}
                for arg in args:
                    if isinstance(arg, torch.Tensor) and hasattr(arg, 'optrace'):
                        merged_optrace.update(arg.optrace)

                merged_optrace[id(result)] = ("map", func, args)
                result.optrace = merged_optrace
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
    if not hasattr(result, 'optrace'):
        result.optrace = {}
    # index edge (edge_type, indicies, orig_arg)
    index_edge = ("index", index, tensor)
    result.optrace[id(result)] = index_edge
    return result


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
        merged_optrace = {}
        for arg in args:
            if isinstance(arg, torch.Tensor) and hasattr(arg, 'optrace'):
                merged_optrace.update(arg.optrace)

        merged_optrace[id(result)] = ("map", func, args)
        result.optrace = merged_optrace
        return result
    return wrapper

    # map_transform(vmap(func))
    # this is wrong: vmap(map_transform(func))
