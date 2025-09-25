#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/DeviceThreadHandles.h>

#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <iostream>
#include <cudss.h>


using namespace std;

static void HandlecusolverError(cusolverStatus_t err, int line) {
    if (err != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "ERROR: %d in %s at line %d, (error-code %d)\n", err, __FILE__, line, err);
        fflush(stdout);
        exit(-1);
    }
}


// template <typename index_t, typename value_t>
torch::Tensor cusolvesp_impl(torch::Tensor A, torch::Tensor b) {
    // Assert A is CSR matrix and b is 1D tensor
    TORCH_CHECK(A.is_sparse_csr(), "A must be a CSR matrix");
    TORCH_CHECK(b.dim() == 1, "b must be a 1D tensor");
    TORCH_CHECK(A.dtype() == b.dtype(), "A and b must have the same dtype");

    int *crow_in, *col_in;
    torch::Tensor crow = A.crow_indices();
    torch::Tensor col = A.col_indices();
    if (crow.dtype() != torch::kInt32) {
        crow = crow.to(torch::kInt32);
        col = col.to(torch::kInt32);
    }
    crow_in = crow.data<int>();
    col_in = col.data<int>();

    int nnz = A._nnz();
    int m = A.size(0);
    torch::Tensor x = torch::empty_like(b);

    int singularity_out;

    cusolverSpHandle_t handle;
    HandlecusolverError(cusolverSpCreate(&handle), __LINE__);
    cusparseMatDescr_t desc;
    cusparseCreateMatDescr(&desc);
    cusparseSetMatDiagType(desc, CUSPARSE_DIAG_TYPE_NON_UNIT);

    torch::Tensor values = A.values();
    if (values.type().scalarType() == torch::ScalarType::Double) 
{            double *csrValA = A.values().data<double>();
            double *b_in = b.data<double>();
            double *x_out = x.data<double>();
            HandlecusolverError(
                cusolverSpDcsrlsvchol(handle, m, nnz, desc, csrValA, crow_in,
                                    col_in, b_in, 1, 3, x_out, &singularity_out),
                __LINE__);
} else if (values.type().scalarType() == torch::ScalarType::Float) {
            float *csrValA = A.values().data<float>();
            float *b_in = b.data<float>();
            float *x_out = x.data<float>();
            HandlecusolverError(
                cusolverSpScsrlsvchol(handle, m, nnz, desc, csrValA, crow_in,
                                    col_in, b_in, 1, 3, x_out, &singularity_out),
                __LINE__);
    }

    HandlecusolverError(cusolverSpDestroy(handle), __LINE__);
    return x;
}


class CuDirectSparseSolver {
    private:
        cudssData_t cudss_data;
        cudssHandle_t handle;
        int called_count = 0;
    public:
        CuDirectSparseSolver() {
            cudssCreate(&handle);
            auto stream = c10::cuda::getCurrentCUDAStream();
            cudssSetStream(handle, stream);
            cudssDataCreate(handle, &cudss_data);
        }

        ~CuDirectSparseSolver() {
            cudssDataDestroy(handle, cudss_data);
            cudssDestroy(handle);
        }

        torch::Tensor operator()(torch::Tensor A, torch::Tensor b) {
            // std::cout << "cudss called_count: " << called_count << std::endl;
            TORCH_CHECK(A.is_sparse_csr(), "A must be a CSR matrix");
            // TORCH_CHECK(b.dim() == 1, "b must be a 1D tensor");
            if (b.dim() == 2) {
                b = b.squeeze(1);
            }

            TORCH_CHECK(A.dtype() == b.dtype(), "A and b must have the same dtype");
        
            // Device pointers and scalar shape parameters, matrix properties
            
            torch::Tensor crow = A.crow_indices();
            torch::Tensor col = A.col_indices();
            if (crow.dtype() != torch::kInt32) {
                crow = crow.to(torch::kInt32);
                col = col.to(torch::kInt32);
            }
            int*    rowOffsets = crow.data<int>();
            int*    colIndices = col.data<int>();
            torch::Tensor values     = A.values();
            torch::Tensor x = torch::empty_like(b);
            //---------------------------------------------------------------------------------
            // cuDSS data structures and handle initialization
            cudssConfig_t             config;
            cudssMatrix_t             b_mt;
            cudssMatrix_t             A_mt;
            cudssMatrix_t             x_mt;
        
        
            cudssConfigCreate(&config);
            // cudssAlgType_t reorder_alg = CUDSS_ALG_3;
            // cudssConfigSet(config, CUDSS_CONFIG_REORDERING_ALG, &reorder_alg, sizeof(cudssAlgType_t));

            if (values.type().scalarType() == torch::ScalarType::Double) {
                double* values_ptr = values.data<double>();
                double* b_ptr = b.data<double>();
                double* x_ptr = x.data<double>();
                cudssMatrixCreateDn(&b_mt, b.size(0), 1, b.size(0), b_ptr, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR);
                cudssMatrixCreateDn(&x_mt, x.size(0), 1, x.size(0), x_ptr, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR);
                cudssMatrixCreateCsr(&A_mt, A.size(0), A.size(1),  A._nnz(), rowOffsets, rowOffsets + crow.size(0), colIndices, values_ptr, CUDA_R_32I, CUDA_R_64F, CUDSS_MTYPE_SPD, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO);
            } else if (values.type().scalarType() == torch::ScalarType::Float) {
                float* values_ptr = values.data<float>();
                float* b_ptr = b.data<float>();
                float* x_ptr = x.data<float>();
                cudssMatrixCreateDn(&b_mt, b.size(0), 1, b.size(0), b_ptr, CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR);
                cudssMatrixCreateDn(&x_mt, x.size(0), 1, x.size(0), x_ptr, CUDA_R_32F, CUDSS_LAYOUT_COL_MAJOR);
                cudssMatrixCreateCsr(&A_mt, A.size(0), A.size(1),  A._nnz(), rowOffsets, rowOffsets + crow.size(0), colIndices, values_ptr, CUDA_R_32I, CUDA_R_32F, CUDSS_MTYPE_SPD, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO);
            }
            //---------------------------------------------------------------------------------
            if (called_count == 0) {
                // Reordering & symbolic factorization
                torch::profiler::impl::cudaStubs()->rangePush("Reordering & symbolic factorization");
                cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, cudss_data, A_mt, x_mt, b_mt);
                // https://docs.nvidia.com/cuda/cudss/types.html?highlight=cudss_data_perm_row
                torch::profiler::impl::cudaStubs()->rangePop();
            }
            //---------------------------------------------------------------------------------
            // Numerical factorization
            torch::profiler::impl::cudaStubs()->rangePush("Numerical factorization");
            cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, cudss_data, A_mt, x_mt, b_mt);
        
            // Retrieve nnz for L matrix
        
            size_t L_nnz = 0;
            size_t sizeWritten = 0;
            cudssStatus_t status;
        
            status = cudssDataGet(handle, cudss_data, CUDSS_DATA_LU_NNZ, &L_nnz, sizeof(L_nnz), &sizeWritten);
            if (status != CUDSS_STATUS_SUCCESS) {
                std::cerr << "Error retrieving L matrix nnz: " << status << std::endl;
                // Handle error appropriately
            }
        
            int64_t A_nrows = 0, A_ncols = 0;
            int64_t A_nnz = 0;
            void *A_rowStart = nullptr, *A_rowEnd = nullptr, *A_colIndices = nullptr, *A_values = nullptr;
            cudaDataType_t A_indexType, A_valueType;
            cudssMatrixType_t A_mtype;
            cudssMatrixViewType_t A_mview;
            cudssIndexBase_t A_indexBase;
        
            status = cudssMatrixGetCsr(A_mt, &A_nrows, &A_ncols, &A_nnz, &A_rowStart, &A_rowEnd, &A_colIndices, &A_values, 
                                    &A_indexType, &A_valueType, &A_mtype, &A_mview, &A_indexBase);
            if (status != CUDSS_STATUS_SUCCESS) {
                std::cerr << "Error retrieving A matrix CSR info: " << status << std::endl;
                // Handle error appropriately
            }
        
            if (A_nnz <= 0) {
                std::cerr << "Original matrix A has zero or negative nnz." << std::endl;
                // Handle error appropriately
            }
        
            double fill_in_factor = static_cast<double>(L_nnz) / static_cast<double>(A_nnz);
            std::cout << "Fill-in factor: " << fill_in_factor << std::endl;
        
        
            torch::profiler::impl::cudaStubs()->rangePop();
        
            //---------------------------------------------------------------------------------
            // Solving the system
            torch::profiler::impl::cudaStubs()->rangePush("Solving the system");
            cudssExecute(handle, CUDSS_PHASE_SOLVE, config, cudss_data, A_mt, x_mt, b_mt);
            torch::profiler::impl::cudaStubs()->rangePop();
        
            //---------------------------------------------------------------------------------
            // (optional) Extra data can be retrieved from the cudssData_t object
            // For example, diagonal of the factorized matrix or the reordering permutation
        
            //---------------------------------------------------------------------------------
            // Destroy the opaque objects
            cudssConfigDestroy(config);
            // cudssDataDestroy(handle, cudss_data);
            cudssMatrixDestroy(A_mt);
            cudssMatrixDestroy(x_mt);
            cudssMatrixDestroy(b_mt);
            // cudssDestroy(handle);
        
            called_count++;
            return x;
        }
};
    
    
// Define the Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<CuDirectSparseSolver>(m, "CuDirectSparseSolver")
        .def(py::init<>())
        .def("__call__", &CuDirectSparseSolver::operator(), "Solve Ax = b using cuDSS");
    m.def("cusolvesp", &cusolvesp_impl, "Solve Ax = b using cuSolverSP");
}

