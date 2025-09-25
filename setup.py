import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

VERSION = "0.1"

def readme():
    """Read the README.md file for long description"""
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return "PyTorch implementation of BA"

# Check if CUDSS should be enabled
USE_CUDSS = os.environ.get("USE_CUDSS", "1").lower() in ("1", "true", "yes", "y")
# Optionally get CUDSS directory from environment variable
CUDSS_DIR = os.environ.get("CUDSS_DIR", "")

if __name__ == '__main__':
    # Common extensions
    ext_modules = [
        CppExtension(
            'bae.sparse.bsr', 
            [os.path.join('bae', 'sparse', 'sparse_op_cpp.cpp')]
        ),
        CUDAExtension(
            'bae.sparse.bsr_cuda', 
            [
                os.path.join('bae', 'sparse', 'sparse_op_cuda.cpp'),
                os.path.join('bae', 'sparse', 'sparse_op_cuda_kernel.cu')
            ]
        ),
        CUDAExtension(
            'bae.sparse.spgemm', 
            [os.path.join('bae', 'sparse', 'cusparse_wrapper.cpp')]
        ),
        CUDAExtension(
            'bae.sparse.conversion', 
            [os.path.join('bae', 'sparse', 'sparse_conversion.cu')]
        ),
    ]
    
    # Add CUDSS-dependent extension conditionally
    if USE_CUDSS:
        libraries = ['cusolver', 'cusparse', 'cudss']
        extra_compile_args = {
            'nvcc': [
                '-lcusolver',
                '-lcusparse',
                '-lcudss',
            ]
        }
        
        # Add CUDSS directory paths if specified
        if CUDSS_DIR:
            extra_compile_args['nvcc'].extend([
                f'-I{CUDSS_DIR}/include',
                f'-L{CUDSS_DIR}/lib',
                f'-Xlinker={CUDSS_DIR}/lib/libcudss_static.a',
            ])
            
        ext_modules.append(
            CUDAExtension(
                'bae.sparse.solve',
                [os.path.join('bae', 'sparse', 'sparse_cusolve.cu')],
                libraries=libraries,
                extra_compile_args=extra_compile_args,
                include_dirs=['/usr/include/libcudss/12/'],
                library_dirs=['/usr/lib/x86_64-linux-gnu/libcudss/12/']
            )
        )

    setup(
        name = 'bae',
        version = VERSION,
        description = 'PyTorch implementation of BA',
        long_description = readme(),
        long_description_content_type = "text/markdown",
        python_requires = ">=3.8",
        install_requires=[
            'torch',
            'torchvision',
            'warp-lang',
        ],
        packages=find_packages(exclude=['./ba_example.py', 
                                        './setup.py', 
                                        './README.md',
                                        './data',
                                        './1dsfm_bal',
                                        './bal_data',
                                        './colmap_helpers',
                                        './datapipes',
                                        './examples',
                                        './tests',]),
        ext_modules=ext_modules,
        cmdclass={'build_ext': BuildExtension}
    )
