#include <iostream>
#include <torch/extension.h>
#include <ATen/ops/mm_native.h>
#include <pybind11/pybind11.h>
#include <ATen/Dispatch.h>
#include <tuple>

#include <ATen/Config.h>
#if AT_MKL_ENABLED() && (!defined(_WIN32))
#define AT_USE_MKL_SPARSE() 1
#else
#define AT_USE_MKL_SPARSE() 0
#endif

#if defined(USE_ROCM)
#include <rocsparse/rocsparse.h>
#include <hip/hip_runtime_api.h>

#define HIP_CHECK(stat)                                                  \
  {                                                                      \
    if (stat != hipSuccess)                                              \
    {                                                                    \
      std::cerr << "Error: hip error in line " << __LINE__ << std::endl; \
    }                                                                    \
  }

#define ROCSPARSE_CHECK(stat)                                                  \
  {                                                                            \
    if (stat != rocsparse_status_success)                                      \
    {                                                                          \
      std::cerr << "Error: rocsparse error in line " << __LINE__ << std::endl; \
    }                                                                          \
  }


torch::Tensor sparse_bsr_mm_rocm(const torch::Tensor a, const torch::Tensor b)
{
  rocsparse_int m = a.size(0);
  rocsparse_int n = b.size(1);
  TORCH_CHECK(a.size(1) == b.size(0), "matrices are not aligned");
  rocsparse_int k = a.size(1);
  TORCH_CHECK(a.values().size(1) == a.values().size(2), "a must have square blocks");
  TORCH_CHECK(b.values().size(1) == b.values().size(2), "b must have square blocks");
  TORCH_CHECK(a.values().size(2) == b.values().size(1), "a and b must have compatible block sizes");
  rocsparse_int block_dim = a.values().size(2);
  rocsparse_int nnzb_A = a.values().size(0);
  rocsparse_int nnzb_B = b.values().size(0);
  rocsparse_int nnzb_D = 0;
  rocsparse_int mb = m / block_dim;
  rocsparse_int nb = n / block_dim;
  rocsparse_int kb = k / block_dim;

  torch::Tensor crow_A = a.crow_indices();
  torch::Tensor col_A = a.col_indices();
  torch::Tensor crow_B = b.crow_indices();
  torch::Tensor col_B = b.col_indices();
  if (crow_A.dtype() != torch::kInt32)
  {
    crow_A = crow_A.to(torch::kInt32);
    col_A = col_A.to(torch::kInt32);
  }
  if (crow_B.dtype() != torch::kInt32)
  {
    crow_B = crow_B.to(torch::kInt32);
    col_B = col_B.to(torch::kInt32);
  }
  rocsparse_int *bsr_row_ptr_A = crow_A.data<rocsparse_int>();
  rocsparse_int *bsr_col_ind_A = col_A.data<rocsparse_int>();
  rocsparse_int *bsr_row_ptr_B = crow_B.data<rocsparse_int>();
  rocsparse_int *bsr_col_ind_B = col_B.data<rocsparse_int>();
  rocsparse_int *bsr_row_ptr_D = nullptr;
  rocsparse_int *bsr_col_ind_D = nullptr;

  // rocSPARSE handle
  rocsparse_handle handle;
  ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

  // Initialize scalar multipliers
  float alpha = 1.0f;
  float beta = 0.0f;

  // Create matrix descriptors
  rocsparse_mat_descr descr_A;
  rocsparse_mat_descr descr_B;
  rocsparse_mat_descr descr_C;
  rocsparse_mat_descr descr_D;

  rocsparse_create_mat_descr(&descr_A);
  rocsparse_create_mat_descr(&descr_B);
  rocsparse_create_mat_descr(&descr_C);
  rocsparse_create_mat_descr(&descr_D);

  // Create matrix info structure
  rocsparse_mat_info info_C;
  rocsparse_create_mat_info(&info_C);

  // Set pointer mode
  rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host);

  // Query rocsparse for the required buffer size
  size_t buffer_size;

  rocsparse_sbsrgemm_buffer_size(handle,
                                 rocsparse_direction_row,
                                 rocsparse_operation_none,
                                 rocsparse_operation_none,
                                 mb,
                                 nb,
                                 kb,
                                 block_dim,
                                 &alpha,
                                 descr_A,
                                 nnzb_A,
                                 bsr_row_ptr_A,
                                 bsr_col_ind_A,
                                 descr_B,
                                 nnzb_B,
                                 bsr_row_ptr_B,
                                 bsr_col_ind_B,
                                 &beta,
                                 descr_D,
                                 nnzb_D,
                                 bsr_row_ptr_D,
                                 bsr_col_ind_D,
                                 info_C,
                                 &buffer_size);

  // Allocate buffer
  void *buffer;
  hipMalloc(&buffer, buffer_size);

  // Obtain number of total non-zero block entries in C and block row pointers of C
  torch::TensorOptions int_tensor_option = torch::TensorOptions().dtype(torch::kInt32).device(crow_A.device());
  rocsparse_int nnzb_C;
  torch::Tensor crow_C = torch::empty({mb + 1}, int_tensor_option);
  rocsparse_int *bsr_row_ptr_C = crow_C.data<rocsparse_int>();

  rocsparse_bsrgemm_nnzb(handle,
                         rocsparse_direction_row,
                         rocsparse_operation_none,
                         rocsparse_operation_none,
                         mb,
                         nb,
                         kb,
                         block_dim,
                         descr_A,
                         nnzb_A,
                         bsr_row_ptr_A,
                         bsr_col_ind_A,
                         descr_B,
                         nnzb_B,
                         bsr_row_ptr_B,
                         bsr_col_ind_B,
                         descr_D,
                         nnzb_D,
                         bsr_row_ptr_D,
                         bsr_col_ind_D,
                         descr_C,
                         bsr_row_ptr_C,
                         &nnzb_C,
                         info_C,
                         buffer);

  // Compute block column indices and values of C
  torch::Tensor col_C = torch::empty({nnzb_C}, int_tensor_option);
  torch::Tensor val_C = torch::empty({nnzb_C, block_dim, block_dim}, torch::TensorOptions().dtype(torch::kFloat32).device(crow_A.device()));
  rocsparse_int *bsr_col_ind_C = col_C.data<rocsparse_int>();

  float *bsr_val_A = a.values().data<float>();
  float *bsr_val_B = b.values().data<float>();
  float *bsr_val_C = val_C.data<float>();
  float *bsr_val_D = nullptr;

  rocsparse_sbsrgemm(handle,
                     rocsparse_direction_row,
                     rocsparse_operation_none,
                     rocsparse_operation_none,
                     mb,
                     nb,
                     kb,
                     block_dim,
                     &alpha,
                     descr_A,
                     nnzb_A,
                     bsr_val_A,
                     bsr_row_ptr_A,
                     bsr_col_ind_A,
                     descr_B,
                     nnzb_B,
                     bsr_val_B,
                     bsr_row_ptr_B,
                     bsr_col_ind_B,
                     &beta,
                     descr_D,
                     nnzb_D,
                     bsr_val_D,
                     bsr_row_ptr_D,
                     bsr_col_ind_D,
                     descr_C,
                     bsr_val_C,
                     bsr_row_ptr_C,
                     bsr_col_ind_C,
                     info_C,
                     buffer);
  ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_A));
  ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_B));
  ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_C));
  ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_D));
  ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
  hipFree(buffer);
  return at::_sparse_compressed_tensor_unsafe(crow_C, col_C, val_C, {m, n}, at::TensorOptions().dtype(val_C.dtype()).device(val_C.device()).layout(at::kSparseBsr));
}


#endif

#define CHECK_CUSPARSE(call)                                                   \
{                                                                              \
    cusparseStatus_t err;                                                      \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS)                             \
    {                                                                          \
        fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__);   \
        cudaError_t cuda_err = cudaGetLastError();                             \
        if (cuda_err != cudaSuccess)                                           \
        {                                                                      \
            fprintf(stderr, "  CUDA error \"%s\" also detected\n",             \
                    cudaGetErrorString(cuda_err));                             \
        }                                                                      \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

torch::Tensor sparse_bsr_mm_cuda(const torch::Tensor, const torch::Tensor);

torch::Tensor sparse_bsr_csr_mm(const torch::Tensor &a, const torch::Tensor &b)
{

  if (a.layout() == at::kSparseBsr && b.layout() == at::kSparseBsc)
  {
    CHECK_CUDA(a);
    CHECK_CUDA(b);
#if defined(USE_ROCM)
    return sparse_bsr_mm_rocm(a, b);
#else
    return sparse_bsr_mm_cuda(a, b);
#endif
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

TORCH_LIBRARY_IMPL(aten, SparseCsrCUDA, m)
{
  // m.impl("mm", sparse_bsr_csr_mm);
}

/*
the file that calls the cuBLAS API for matrix multiplication is located at aten/src/ATen/native/cuda/Blas.cpp. In this file, there is a function called matmul_kernel that dispatches different cuBLAS functions based on the input tensor shapes and data types. For example, if the input tensors are 2-dimensional and have floating-point values, the function will call cublasSgemm or cublasDgemm for single-precision or double-precision arithmetic, respectively.

the file that calls the MKL API for matrix multiplication is located at aten/src/ATen/native/LinearAlgebra.cpp. In this file, there is a function called bmm_out_or_baddbmm_ that dispatches different MKL functions based on the input tensor shapes and data types. For example, if the input tensors are 2-dimensional and have floating-point values, the function will call cblas_sgemm or cblas_dgemm for single-precision or double-precision arithmetic, respectively.

Blas: same file but in different function
at::blas::gemm
*/