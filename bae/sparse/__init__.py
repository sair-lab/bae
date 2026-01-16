from .bsr import *
from .bsr_cuda import *
from .py_ops import *
try:
    from .solve import *
except ImportError:
    # `bae.sparse.solve` depends on NVIDIA cuDSS. Some environments ship cuDSS built
    # against a newer CUDA/cuBLAS (e.g. `libcublas.so.13`), which makes importing
    # this package fail even if you don't use the direct solver. Keep the rest of
    # the sparse ops usable and let callers opt into `bae.sparse.solve` explicitly.
    pass
from .conversion import *
from .warp_wrappers import *
