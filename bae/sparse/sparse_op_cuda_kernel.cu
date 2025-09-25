#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <vector>

#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/adjacent_difference.h>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

template <typename index_t>
__global__ void scan_symbol_kernel(
    const index_t *coo_indices_ptr,
    const index_t *crow_indices_ptr,
    const index_t *ccol_indices_ptr,
    const index_t *row_indices_ptr,
    const index_t *col_indices_ptr,
    int64_t sm,
    int64_t sp,
    index_t *out_data_ptr)
{
  int64_t count;
  int64_t max_partial = sp;
  int nnza = crow_indices_ptr[sm];
  index_t col_indices_ptr_k1;
  for (int k1 = blockIdx.x * blockDim.x + threadIdx.x; k1 < nnza; k1 += blockDim.x * gridDim.x)
  {
    int64_t i = coo_indices_ptr[k1];
    // int64_t i = blockIdx.x; // each block is responsible for a row
    // TODO: Move crow indices to shared memory
    // for (auto k1 = crow_indices_ptr[i] + threadIdx.x; k1 < crow_indices_ptr[i + 1]; k1 = k1 + blockDim.x) // TODO: allocate cache in case of more than one iter
    count = 0;
    col_indices_ptr_k1 = col_indices_ptr[k1];
    for (int64_t j = 0; j < sp; j++)
    {
      index_t k2 = ccol_indices_ptr[j];
      index_t col_end = ccol_indices_ptr[j + 1];
      if (k2 == col_end)
        continue;
      while (row_indices_ptr[k2] < col_indices_ptr_k1 && k2 < col_end - 1)
      {
        k2 += 1;
      }
      if (row_indices_ptr[k2] == col_indices_ptr_k1)
      {
        int global_pos = k1 * max_partial + count;
        out_data_ptr[nnza + global_pos * 4 + 0] = k1;
        out_data_ptr[nnza + global_pos * 4 + 1] = k2;
        out_data_ptr[nnza + global_pos * 4 + 2] = i;
        out_data_ptr[nnza + global_pos * 4 + 3] = j;
        count += 1;
      }
    }
    out_data_ptr[k1] = count;
  }

  __syncthreads();
}

template <typename index_t, typename count_t>
__global__ void flatten_var_len(index_t *in_data_ptr, count_t stride,
                                index_t *offsets, index_t *elements_per_row, index_t *out_data_ptr, size_t size_per_cell)
{
  // use memcpy to copy data from in_data_ptr to out_data_ptr
  // in_data_ptr is a 2D array of size num_rows * (stride * size_per_cell)
  // out_data_ptr is a 2D array of size (total_elements * size_per_cell)
  // offsets is a 1D array of size num_rows
  // elements_per_row is a 1D array of size num_rows
  // stride is the number of max elements per row
  // size_per_cell is the size of each element in bytes
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  int offset = offsets[i];
  int num_elements = elements_per_row[i];
  memcpy(out_data_ptr + offset * size_per_cell, in_data_ptr + i * stride * size_per_cell, num_elements * size_per_cell * sizeof(index_t));
}

torch::Tensor sparse_bsr_mm_cuda(const torch::Tensor bsr, const torch::Tensor bsc)
{
  auto crow_indices = bsr.crow_indices();
  auto col_indices = bsr.col_indices();
  auto csr_values = bsr.values();

  auto ccol_indices = bsc.ccol_indices();
  auto row_indices = bsc.row_indices();
  auto csc_values = bsc.values();
  CHECK_CONTIGUOUS(crow_indices);
  CHECK_CONTIGUOUS(col_indices);
  CHECK_CONTIGUOUS(ccol_indices);
  CHECK_CONTIGUOUS(row_indices);

  TORCH_CHECK_EQ(bsr.ndimension(), 2);
  TORCH_CHECK_EQ(bsc.ndimension(), 2);

  auto m = bsr.size(-2);
  auto n = bsr.size(-1);
  auto p = bsc.size(-1);
  int dm;
  int dn;
  int dp;
  if (bsr.layout() == at::kSparseCsr && bsc.layout() == at::kSparseCsc)
  {
    TORCH_CHECK_EQ(csr_values.ndimension(), 1);
    TORCH_CHECK_EQ(csc_values.ndimension(), 1);
    dm = 1;
    dn = 1;
    dp = 1;
  }
  else {
    dm = csr_values.size(-2);
    dn = csr_values.size(-1);
    dp = csc_values.size(-1);
  }
  auto sm = m / dm;
  auto sn = n / dn;
  auto sp = p / dp;
  TORCH_CHECK_EQ(dm * sm, m);
  TORCH_CHECK_EQ(dn * sn, n);
  TORCH_CHECK_EQ(dp * sp, p);

  torch::Tensor cooa = at::_convert_indices_from_csr_to_coo(crow_indices, col_indices, crow_indices.dtype() == at::kInt, false);
  // std::cout << "cooa: " << cooa << std::endl;
  torch::Tensor sources;
  torch::Tensor coo_indices;
  torch::Tensor index;
  int nuijs;
  AT_DISPATCH_INDEX_TYPES(
      crow_indices.scalar_type(),
      "bsr_mm_crow_indices",
      [&]() { // maybe useful: CppTypeToScalarType
        auto nnza = col_indices.size(0);
        if (nnza == 0)
        {
          int total_partials = 0;
          nuijs = 0;
          sources = torch::empty({total_partials, 4}, torch::TensorOptions(crow_indices.dtype()).device(crow_indices.device()));
          index = torch::empty({total_partials}, torch::TensorOptions(crow_indices.dtype()).device(crow_indices.device()));
          coo_indices = torch::empty({0, 2}, torch::TensorOptions(crow_indices.dtype()).device(crow_indices.device()));
          return;
        }
        index_t *out_data_ptr;
        cudaMalloc(&out_data_ptr, sizeof(index_t) * 5 * (nnza * sp));
        int numSMs;
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0); // Perform SAXPY on 1M elements saxpy<<<32*numSMs, 256>>>(1 << 20, 2.0, x, y);
        scan_symbol_kernel<index_t><<<32 * numSMs, 256>>>(
            cooa.data<index_t>(),
            crow_indices.data<index_t>(),
            ccol_indices.data<index_t>(),
            row_indices.data<index_t>(),
            col_indices.data<index_t>(),
            sm, sp, out_data_ptr);
        cudaDeviceSynchronize();
        // Determine temporary device storage requirements
        void *d_temp_storage = NULL;
        index_t *offsets;
        cudaMalloc(&offsets, nnza * sizeof(index_t));
        thrust::exclusive_scan(thrust::device, out_data_ptr, out_data_ptr + nnza, offsets);
        // auto total_partials = out_data_ptr[num_items - 1] + offsets[num_items - 1];
        index_t total_partials = 0;
        index_t tmp;
        cudaMemcpy(&total_partials, out_data_ptr + nnza - 1, sizeof(index_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&tmp, offsets + nnza - 1, sizeof(index_t), cudaMemcpyDeviceToHost);
        total_partials += tmp;
        // std::cout << "total_partials: " << total_partials << std::endl;

        // Allocate kkijs
        sources = torch::empty({total_partials, 4}, torch::TensorOptions(crow_indices.dtype()).device(crow_indices.device()));
        index = torch::empty({total_partials}, torch::TensorOptions(crow_indices.dtype()).device(crow_indices.device()));
        // torch::zeros({total_partials, 4}, torch::TensorOptions(index_t).device(crow_indices.device()));  // TODO
        if (total_partials == 0)
        {
          coo_indices = torch::empty({0, 2}, torch::TensorOptions(crow_indices.dtype()).device(crow_indices.device()));
          return;
        }
        // Copy data to sources
        // std::cout << "!!!.....!!!!" << std::endl;
        flatten_var_len<index_t, int64_t><<<nnza, 1>>>(out_data_ptr + nnza,
                                                       sp,
                                                       offsets,
                                                       out_data_ptr,
                                                       sources.data<index_t>(),
                                                       4);
        cudaDeviceSynchronize();
        // std::cout << "111111" << std::endl;
        // assign_rank<index_t, int64_t><<<1, total_partials>>>(sources.data<index_t>(), sm, sp, total_partials, coo_indices.data<index_t>(), nuijs_ptr, index.data<index_t>());

        torch::Tensor ijs = sources.index({torch::indexing::Slice(0, total_partials), 2}) * sp + sources.index({torch::indexing::Slice(0, total_partials), 3});
        torch::Tensor partial_indices = torch::arange(0, total_partials, torch::TensorOptions().dtype(crow_indices.dtype()).device(crow_indices.device()));
        // sort by key
        // https://nvidia.github.io/cccl/thrust/api/groups/group__sorting.html#function-sort-by-key
        thrust::sort_by_key(thrust::device, ijs.data<index_t>(), ijs.data<index_t>() + total_partials, partial_indices.data<index_t>());

        // struct equals : public thrust::binary_function<index_t,index_t,index_t>
        // {
        //   index_t operator()(index_t x, index_t y) { return !(x ^ y); }
        // };
        // adj diff
        thrust::adjacent_difference(thrust::device, ijs.data<index_t>(), ijs.data<index_t>() + total_partials, index.data<index_t>());
        index.clamp_(0, 1);
        index[0] = 1;

        torch::Tensor uijs = ijs.masked_select(index.to(torch::kBool)); // TODO: check if casting is necessary
        coo_indices = torch::empty({uijs.size(0), 2}, torch::TensorOptions(crow_indices.dtype()).device(crow_indices.device()));
        coo_indices.index({torch::indexing::Slice(0, uijs.size(0)), 0}) = uijs / sp;
        coo_indices.index({torch::indexing::Slice(0, uijs.size(0)), 1}) = uijs % sp;

        //
        // https://nvidia.github.io/cccl/thrust/api/groups/group__prefixsums.html#function-inclusive-scan
        thrust::inclusive_scan(thrust::device, index.data<index_t>(), index.data<index_t>() + total_partials, index.data<index_t>());
        nuijs = index[total_partials - 1].item<int>();
        // std::cout << "nuijs: " << nuijs << std::endl;
        index -= 1; // TODO: should be more efficient
        // TORCH_CHECK_EQ(nuijs, uijs.size(0));
        // index[partial_indices] = index;
        index = torch::empty_like(index).scatter_(0, partial_indices, index);

        cudaFree(out_data_ptr);
        cudaFree(offsets);
      });
  // std::cout << "coo_indices[..., 0]: " << coo_indices.index({"...", 0}) << std::endl;
  // print coo_indices using tensor accessor
  // for (int j = 0; j < sp; j++js)) {
  //   std::cout << coo_indices[i][0].item<int>() << " " << coo_indices[i][1].item<int>() << std::endl;
  // }
  auto prod = torch::bmm(csr_values.index({sources.index({"...", 0})}), csc_values.index({sources.index({"...", 1})}));
  auto reduced = torch::zeros({nuijs, dm, dp}, prod.options());
  reduced.scatter_add_(0, index.index({"...", torch::indexing::None, torch::indexing::None}).expand_as(prod), prod);
  auto row_res = coo_indices.index({"...", 0}).contiguous();
  auto col_res = coo_indices.index({"...", 1}).contiguous();
  // std::cout << "sm, sp: " << sm << " " << sp << std::endl;
  // std::cout << "row_res: " << row_res << std::endl;
  // std::cout << "col_res: " << col_res << std::endl;
  auto crow_res = at::_convert_indices_from_coo_to_csr(row_res, sm, row_res.dtype() == at::kInt);
  // auto ccol_res = at::_convert_indices_from_coo_to_csr(col_res, sm, col_res.dtype() == at::kInt);
  // std::cout << "crow_res: " << crow_res << std::endl;
  // std::cout << "ccol_res: " << ccol_res << std::endl;
  // std::cout << "reduced: " << reduced << std::endl;
  // at::IntArrayRef();
  // int64_t *size = (int64_t *)malloc(2 * sizeof(int64_t));
  // size[0] = sm;
  // size[1] = sp;
  // auto dummy_coo = torch::sparse_coo_tensor(coo_indices.mT(), torch::zeros({nuijs}, row_res.options()), {sm, sp}, at::TensorOptions().dtype(row_res.dtype()).device(row_res.device()).layout(at::kSparse));
  // auto crow_res = dummy_coo.to_sparse_csr().crow_indices().to(reduced.device());
  return at::_sparse_compressed_tensor_unsafe(crow_res.to(reduced.device()), col_res.to(reduced.device()), reduced.to(reduced.device()), {m, p}, at::TensorOptions().dtype(reduced.dtype()).device(reduced.device()).layout(at::kSparseBsr));
}
