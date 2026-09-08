#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/DeviceThreadHandles.h>

#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <iostream>
#include <cudss.h>


// cuDSS 0.8 introduced its own datatypes and separate CSR offset/index types.
#if CUDSS_VERSION_MAJOR > 0 || CUDSS_VERSION_MINOR >= 8
using BaeCudssDataType = cudssDataType_t;
constexpr auto BAE_CUDSS_R_32F = CUDSS_R_32F;
constexpr auto BAE_CUDSS_R_64F = CUDSS_R_64F;
#else
using BaeCudssDataType = cudaDataType_t;
constexpr auto BAE_CUDSS_R_32F = CUDA_R_32F;
constexpr auto BAE_CUDSS_R_64F = CUDA_R_64F;
#endif

static cudssStatus_t createCsrMatrix(
    cudssMatrix_t* matrix, int64_t nrows, int64_t ncols, int64_t nnz,
    int* rowOffsets, int* colIndices, void* values, BaeCudssDataType valueType) {
    return cudssMatrixCreateCsr(
        matrix, nrows, ncols, nnz, rowOffsets, nullptr, colIndices, values,
#if CUDSS_VERSION_MAJOR > 0 || CUDSS_VERSION_MINOR >= 8
        CUDSS_R_32I, CUDSS_R_32I,
#else
        CUDA_R_32I,
#endif
        valueType, CUDSS_MTYPE_SPD, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO);
}

using namespace std;

static void HandlecusolverError(cusolverStatus_t err, int line) {
    if (err != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "ERROR: %d in %s at line %d, (error-code %d)\n", err, __FILE__, line, err);
        fflush(stdout);
        exit(-1);
    }
}

static void HandlecudssError(cudssStatus_t err, const char* call, int line) {
    TORCH_CHECK(
        err == CUDSS_STATUS_SUCCESS,
        "cuDSS error in ", call, " at ", __FILE__, ":", line,
        " (error-code ", static_cast<int>(err), ")");
}

static void WarncudssError(cudssStatus_t err, const char* call, int line) {
    if (err != CUDSS_STATUS_SUCCESS) {
        std::cerr << "cuDSS error in " << call << " at " << __FILE__ << ":"
                  << line << " (error-code " << static_cast<int>(err) << ")"
                  << std::endl;
    }
}

#define HANDLE_CUDSS_ERROR(call) HandlecudssError((call), #call, __LINE__)
#define WARN_CUDSS_ERROR(call) WarncudssError((call), #call, __LINE__)


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
            HANDLE_CUDSS_ERROR(cudssCreate(&handle));
            auto stream = c10::cuda::getCurrentCUDAStream();
            HANDLE_CUDSS_ERROR(cudssSetStream(handle, stream));
            HANDLE_CUDSS_ERROR(cudssDataCreate(handle, &cudss_data));
        }

        ~CuDirectSparseSolver() {
            WARN_CUDSS_ERROR(cudssDataDestroy(handle, cudss_data));
            WARN_CUDSS_ERROR(cudssDestroy(handle));
        }

        torch::Tensor operator()(torch::Tensor A, torch::Tensor b) {
            // std::cout << "cudss called_count: " << called_count << std::endl;
            TORCH_CHECK(A.is_sparse_csr(), "A must be a CSR matrix");
            TORCH_CHECK(
                b.dim() == 1 || (b.dim() == 2 && b.size(1) == 1),
                "b must be a 1D tensor or a 2D column tensor");

            const std::vector<int64_t> rhs_shape = b.sizes().vec();
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
        
        
            HANDLE_CUDSS_ERROR(cudssConfigCreate(&config));
            // cudssAlgType_t reorder_alg = CUDSS_ALG_3;
            // cudssConfigSet(config, CUDSS_CONFIG_REORDERING_ALG, &reorder_alg, sizeof(cudssAlgType_t));
            
            if (values.type().scalarType() == torch::ScalarType::Double) {
                double* values_ptr = values.data<double>();
                double* b_ptr = b.data<double>();
                double* x_ptr = x.data<double>();
                HANDLE_CUDSS_ERROR(cudssMatrixCreateDn(&b_mt, b.size(0), 1, b.size(0), b_ptr, BAE_CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
                HANDLE_CUDSS_ERROR(cudssMatrixCreateDn(&x_mt, x.size(0), 1, x.size(0), x_ptr, BAE_CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
                HANDLE_CUDSS_ERROR(createCsrMatrix(&A_mt, A.size(0), A.size(1), A._nnz(), rowOffsets, colIndices, values_ptr, BAE_CUDSS_R_64F));
            } else if (values.type().scalarType() == torch::ScalarType::Float) {
                float* values_ptr = values.data<float>();
                float* b_ptr = b.data<float>();
                float* x_ptr = x.data<float>();
                HANDLE_CUDSS_ERROR(cudssMatrixCreateDn(&b_mt, b.size(0), 1, b.size(0), b_ptr, BAE_CUDSS_R_32F, CUDSS_LAYOUT_COL_MAJOR));
                HANDLE_CUDSS_ERROR(cudssMatrixCreateDn(&x_mt, x.size(0), 1, x.size(0), x_ptr, BAE_CUDSS_R_32F, CUDSS_LAYOUT_COL_MAJOR));
                HANDLE_CUDSS_ERROR(createCsrMatrix(&A_mt, A.size(0), A.size(1), A._nnz(), rowOffsets, colIndices, values_ptr, BAE_CUDSS_R_32F));
                // https://docs.nvidia.com/cuda/archive/12.9.0/cudss/functions.html#:~:text=the%20dense%20matrix-,NULL%20is%20the%20only%20supported%20value%20as%204%2Darray%20CSR%20is%20not%20supported%20currently,-colIndices
            }
            //---------------------------------------------------------------------------------
            if (called_count == 0) {
                // Reordering & symbolic factorization
                torch::profiler::impl::cudaStubs()->rangePush("Reordering & symbolic factorization");
                HANDLE_CUDSS_ERROR(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, cudss_data, A_mt, x_mt, b_mt));
                // https://docs.nvidia.com/cuda/cudss/types.html?highlight=cudss_data_perm_row
                torch::profiler::impl::cudaStubs()->rangePop();
            }
            //---------------------------------------------------------------------------------
            // Numerical factorization
            torch::profiler::impl::cudaStubs()->rangePush("Numerical factorization");
            HANDLE_CUDSS_ERROR(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, cudss_data, A_mt, x_mt, b_mt));
        
            // Retrieve nnz for L matrix
        
            size_t L_nnz = 0;
            size_t sizeWritten = 0;
        
            HANDLE_CUDSS_ERROR(cudssDataGet(handle, cudss_data, CUDSS_DATA_LU_NNZ, &L_nnz, sizeof(L_nnz), &sizeWritten));
        
            // The input tensor already provides nnz; no matrix metadata query is needed.
            const int64_t A_nnz = A._nnz();

            TORCH_CHECK(A_nnz > 0, "Original matrix A has zero or negative nnz.");
        
            double fill_in_factor = static_cast<double>(L_nnz) / static_cast<double>(A_nnz);
            std::cout << "Fill-in factor: " << fill_in_factor << std::endl;
        
        
            torch::profiler::impl::cudaStubs()->rangePop();
        
            //---------------------------------------------------------------------------------
            // Solving the system
            torch::profiler::impl::cudaStubs()->rangePush("Solving the system");
            HANDLE_CUDSS_ERROR(cudssExecute(handle, CUDSS_PHASE_SOLVE, config, cudss_data, A_mt, x_mt, b_mt));
            torch::profiler::impl::cudaStubs()->rangePop();
        
            //---------------------------------------------------------------------------------
            // (optional) Extra data can be retrieved from the cudssData_t object
            // For example, diagonal of the factorized matrix or the reordering permutation
        
            //---------------------------------------------------------------------------------
            // Destroy the opaque objects
            HANDLE_CUDSS_ERROR(cudssConfigDestroy(config));
            // cudssDataDestroy(handle, cudss_data);
            HANDLE_CUDSS_ERROR(cudssMatrixDestroy(A_mt));
            HANDLE_CUDSS_ERROR(cudssMatrixDestroy(x_mt));
            HANDLE_CUDSS_ERROR(cudssMatrixDestroy(b_mt));
            // cudssDestroy(handle);
        
            called_count++;
            return x.view(rhs_shape);
        }
};
    
    
// Define the Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<CuDirectSparseSolver>(m, "CuDirectSparseSolver")
        .def(py::init<>())
        .def("__call__", &CuDirectSparseSolver::operator(), "Solve Ax = b using cuDSS");
    m.def("cusolvesp", &cusolvesp_impl, "Solve Ax = b using cuSolverSP");
}
