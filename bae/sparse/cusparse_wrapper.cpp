#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/DeviceThreadHandles.h>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpGEMM

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
                                                                               \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
                                                                               \
    }                                                                          \
}

class CuSparse
{
private:
    int called_count = 0;
    cusparseSpMatDescr_t matA = NULL;
    cusparseSpMatDescr_t matB = NULL;
    cusparseSpMatDescr_t matC = NULL;
    cusparseSpGEMMDescr_t spgemmDesc;
    torch::Tensor c_crow;
    torch::Tensor c_col;
    torch::Tensor c_val;
    cudaStream_t current_stream = c10::cuda::getCurrentCUDAStream();
    void *dBuffer4 = NULL;
    void *dBuffer5 = NULL;

public:
    CuSparse()
    {
        CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc))
    }

    ~CuSparse()
    {
        // Destroy the cuSPARSE handle
        CHECK_CUSPARSE( cusparseSpGEMM_destroyDescr(spgemmDesc) )
        if (matA != NULL)
        {
            CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
            matA = NULL;
        }
        if (matB != NULL)
        {
            CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
            matB = NULL;
        }
        if (matC != NULL)
        {
            CHECK_CUSPARSE( cusparseDestroySpMat(matC) )
            matC = NULL;
        }

        if (dBuffer4 != NULL) {
            CHECK_CUDA(cudaFree(dBuffer4))
            dBuffer4 = NULL;
        }

        if (dBuffer5 != NULL) {
            CHECK_CUDA(cudaFree(dBuffer5))
            dBuffer5 = NULL;
        }
    }
    torch::Tensor
    cusparse_spgemm(const torch::Tensor a, const torch::Tensor b)
    {

// Host problem definition
#define A_NUM_ROWS 4 // C compatibility
        const int A_num_rows = a.size(-2);
        const int A_num_cols = a.size(-1);
        const int A_nnz = a._nnz();
        const int B_num_rows = b.size(-2);
        const int B_num_cols = b.size(-1);
        const int B_nnz = b._nnz();

        double alpha = 1.0f;
        double beta = 0.0f;
        cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
        cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
        cudaDataType computeType = CUDA_R_64F;
        //--------------------------------------------------------------------------
        // Device memory management: Allocate and copy A, B
        int *dA_csrOffsets, *dA_columns, *dB_csrOffsets, *dB_columns,
            *dC_csrOffsets, *dC_columns;
        double *dA_values, *dB_values, *dC_values;
        // allocate A
        torch::Tensor a_crow = a.crow_indices();
        torch::Tensor b_crow = b.crow_indices();
        torch::Tensor a_col = a.col_indices();
        torch::Tensor b_col = b.col_indices();
        if (a_crow.dtype() != torch::kInt32)
        {
            a_crow = a_crow.to(torch::kInt32);
            a_col = a_col.to(torch::kInt32);
        }
        if (b_crow.dtype() != torch::kInt32)
        {
            b_crow = b_crow.to(torch::kInt32);
            b_col = b_col.to(torch::kInt32);
        }
        dA_csrOffsets = a_crow.data<int>();
        dA_columns = a_col.data<int>();
        dA_values = a.values().data<double>();
        // allocate B
        dB_csrOffsets = b_crow.data<int>();
        dB_columns = b_col.data<int>();
        dB_values = b.values().data<double>();

        //--------------------------------------------------------------------------
        // CUSPARSE APIs
        cusparseHandle_t handle = at::cuda::getCurrentCUDASparseHandle();
        if (called_count == 0)
        {
            // allocate C offsets
            c_crow = torch::empty({A_num_rows + 1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
            dC_csrOffsets = c_crow.data<int>();
            size_t bufferSize1 = 0;
            size_t bufferSize2 = 0;
            size_t bufferSize3 = 0;
            size_t bufferSize4 = 0;
            size_t bufferSize5 = 0;
            // CHECK_CUSPARSE(cusparseCreate(&handle))
            // CHECK_CUSPARSE(cusparseSetStream(handle, c10::cuda::getCurrentCUDAStream()))
            // Create sparse matrix A in CSR format
            CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                             dA_csrOffsets, dA_columns, dA_values,
                                             CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                             CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))
            CHECK_CUSPARSE(cusparseCreateCsr(&matB, B_num_rows, B_num_cols, B_nnz,
                                             dB_csrOffsets, dB_columns, dB_values,
                                             CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                             CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))
            CHECK_CUSPARSE(cusparseCreateCsr(&matC, A_num_rows, B_num_cols, 0,
                                             NULL, NULL, NULL,
                                             CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                             CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))
            //==========================================================================
            // SpGEMM Computation
            //==========================================================================

            // ask bufferSize1 bytes for external memory
            CHECK_CUSPARSE(
                cusparseSpGEMMreuse_workEstimation(handle, opA, opB, matA, matB, matC,
                                                   CUSPARSE_SPGEMM_DEFAULT,
                                                   spgemmDesc, &bufferSize1, NULL))
            auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
            at::DataPtr dataPtr1 = allocator.allocate(bufferSize1);
            void *dBuffer1 = dataPtr1.get();
            // inspect the matrices A and B to understand the memory requirement for
            // the next step
            CHECK_CUSPARSE(
                cusparseSpGEMMreuse_workEstimation(handle, opA, opB, matA, matB, matC,
                                                   CUSPARSE_SPGEMM_DEFAULT,
                                                   spgemmDesc, &bufferSize1, dBuffer1))
            //--------------------------------------------------------------------------

            CHECK_CUSPARSE(
                cusparseSpGEMMreuse_nnz(handle, opA, opB, matA, matB,
                                        matC, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc,
                                        &bufferSize2, NULL, &bufferSize3, NULL,
                                        &bufferSize4, NULL))
            at::DataPtr dataPtr2 = allocator.allocate(bufferSize2);
            void *dBuffer2 = dataPtr2.get();
            at::DataPtr dataPtr3 = allocator.allocate(bufferSize3);
            void *dBuffer3 = dataPtr3.get();
            CHECK_CUDA(cudaMallocAsync((void **)&dBuffer4, bufferSize4, current_stream))

            CHECK_CUSPARSE(
                cusparseSpGEMMreuse_nnz(handle, opA, opB, matA, matB,
                                        matC, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc,
                                        &bufferSize2, dBuffer2, &bufferSize3, dBuffer3,
                                        &bufferSize4, dBuffer4))
            //--------------------------------------------------------------------------

            // get matrix C non-zero entries C_nnz1
            int64_t C_num_rows1, C_num_cols1, C_nnz1;
            CHECK_CUSPARSE(cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1,
                                                &C_nnz1))
            // allocate matrix C
            c_col = torch::empty({C_nnz1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
            c_val = torch::zeros({C_nnz1}, torch::dtype(c10::ScalarType::Double).device(torch::kCUDA));
            dC_columns = c_col.data<int>();
            dC_values = c_val.data<double>();

            // fill dC_values if needed
            // update matC with the new pointers
            CHECK_CUSPARSE(
                cusparseCsrSetPointers(matC, dC_csrOffsets, dC_columns, dC_values))
            //--------------------------------------------------------------------------

            CHECK_CUSPARSE(
                cusparseSpGEMMreuse_copy(handle, opA, opB, matA, matB, matC,
                                         CUSPARSE_SPGEMM_DEFAULT, spgemmDesc,
                                         &bufferSize5, NULL))
            CHECK_CUDA(cudaMallocAsync((void **)&dBuffer5, bufferSize5, current_stream))
            CHECK_CUSPARSE(
                cusparseSpGEMMreuse_copy(handle, opA, opB, matA, matB, matC,
                                         CUSPARSE_SPGEMM_DEFAULT, spgemmDesc,
                                         &bufferSize5, dBuffer5))
        }
        else
        {
            // update matC with the new pointers
            CHECK_CUSPARSE(
                cusparseCsrSetPointers(matA, dA_csrOffsets, dA_columns, dA_values))
            CHECK_CUSPARSE(
                cusparseCsrSetPointers(matB, dB_csrOffsets, dB_columns, dB_values))
        }
        called_count++;
        //--------------------------------------------------------------------------
        // first run
        CHECK_CUSPARSE(
            cusparseSpGEMMreuse_compute(handle, opA, opB, &alpha, matA, matB, &beta,
                                        matC, computeType, CUSPARSE_SPGEMM_DEFAULT,
                                        spgemmDesc))
        return at::_sparse_compressed_tensor_unsafe(c_crow, c_col, c_val, {A_num_rows, B_num_cols}, at::TensorOptions().dtype(c_val.dtype()).device(c_val.device()).layout(at::kSparseCsr));
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<CuSparse>(m, "CuSparse")
        .def(py::init<>())
        .def("__call__", &CuSparse::cusparse_spgemm, "");
    m.def("convert_indices_from_csr_to_coo", &at::_convert_indices_from_csr_to_coo, "pytorch aten wrapper");
    m.def("convert_indices_from_coo_to_csr", &at::_convert_indices_from_coo_to_csr, "pytorch aten wrapper");
}
