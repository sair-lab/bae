#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/detail/DeviceThreadHandles.h>

#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
    }                                                                          \
}

torch::Tensor gebsr2csr_impl(torch::Tensor A) {
    TORCH_CHECK(A.layout() == torch::kSparseBsr, "A must be a BSR matrix");
    int *bsrRowPtrA, *bsrColIndA;
    torch::Tensor crow_a = A.crow_indices();
    torch::Tensor col_a = A.col_indices();
    if (crow_a.dtype() != torch::kInt32) {
        crow_a = crow_a.to(torch::kInt32);
        col_a = col_a.to(torch::kInt32);
    }
    // std::cout << "gebsr2csr" << std::endl << std::flush;
    bsrRowPtrA = crow_a.data<int>();
    bsrColIndA = col_a.data<int>();

    int m = A.size(-2);
    int n = A.size(-1);
    int rowBlockDim = A.values().size(-2);
    int colBlockDim = A.values().size(-1);
    int mb = m / rowBlockDim;
    int nb = n / colBlockDim;
    int nnzb = col_a.size(0); // number of blocks
    int nnz  = nnzb * rowBlockDim * colBlockDim; // number of elements

    torch::Tensor crow_c = torch::empty({m+1}, crow_a.options());
    int *csrRowPtrC = crow_c.data_ptr<int>();
    torch::Tensor col_c = torch::empty({nnz}, crow_a.options());
    int *csrColIndC = col_c.data_ptr<int>();
    torch::Tensor val_c = torch::empty({nnz}, A.values().options());
    cusparseHandle_t handle = NULL;
    cusparseMatDescr_t descrA = NULL;
    cusparseMatDescr_t descrC = NULL;

    // Create matrix descriptors
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrC));
    auto stream = c10::cuda::getCurrentCUDAStream();
    CHECK_CUSPARSE(cusparseCreate(&handle));
    CHECK_CUSPARSE(cusparseSetStream(handle, stream));
    cusparseDirection_t dir = CUSPARSE_DIRECTION_COLUMN;
    AT_DISPATCH_FLOATING_TYPES(A.values().scalar_type(), "creating_desc", ([&] {
        const scalar_t *bsrValA = A.values().data_ptr<scalar_t>();
        scalar_t *csrValC = val_c.data_ptr<scalar_t>();
        if (std::is_same<scalar_t, float>::value) {
            CHECK_CUSPARSE( cusparseSgebsr2csr(handle, dir, mb, nb,
                descrA,
                (float *)bsrValA, bsrRowPtrA, bsrColIndA,
                rowBlockDim, colBlockDim,
                descrC,
                (float *)csrValC, csrRowPtrC, csrColIndC) )
        } else if (std::is_same<scalar_t, double>::value) {
            CHECK_CUSPARSE( cusparseDgebsr2csr(handle, dir, mb, nb,
                descrA,
                (double *)bsrValA, bsrRowPtrA, bsrColIndA,
                rowBlockDim, colBlockDim,
                descrC,
                (double *)csrValC, csrRowPtrC, csrColIndC) )
        }
    }));

    CHECK_CUSPARSE( cusparseDestroyMatDescr(descrA) );
    CHECK_CUSPARSE( cusparseDestroyMatDescr(descrC) );
    CHECK_CUSPARSE( cusparseDestroy(handle) );
    
    return at::_sparse_compressed_tensor_unsafe(crow_c, col_c, val_c, {m, n}, at::TensorOptions().dtype(val_c.dtype()).device(val_c.device()).layout(at::kSparseCsr));


}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gebsr2csr", &gebsr2csr_impl, "Custom gebsr2csr function");
}
