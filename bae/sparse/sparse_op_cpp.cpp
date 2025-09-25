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

template <typename index_t>
std::tuple<index_t *, int, index_t *, int> scan_symbol(const index_t *crow_indices_ptr,
                                       const index_t *ccol_indices_ptr,
                                       const index_t *row_indices_ptr,
                                       const index_t *col_indices_ptr,
                                       int64_t sm,
                                       int64_t sp)
{
  std::vector<std::tuple<index_t, index_t, index_t, index_t>> kkijs[sm];
  at::parallel_for(0, sm, at::internal::GRAIN_SIZE, [&](int64_t start, int64_t end) {
  // auto start = 0;
  // auto end = sm;
  for (const auto i : c10::irange(start, end))
  {
    for (const auto k1 : c10::irange(crow_indices_ptr[i], crow_indices_ptr[i + 1]))
    {
      for (const auto j : c10::irange(sp))
      {
        index_t k2 = ccol_indices_ptr[j];
        if (k2 == ccol_indices_ptr[j + 1])
          continue;
        while (row_indices_ptr[k2] < col_indices_ptr[k1] && k2 < ccol_indices_ptr[j + 1] - 1)
        {
          k2 += 1;
        }
        if (row_indices_ptr[k2] == col_indices_ptr[k1])
        {
          kkijs[i].push_back(std::make_tuple(k1, k2, i, j));
        }
      }
    }
  }
  });
  // get nnz
  auto nnz = 0;
  auto nuijs = 0;
  for (const auto i : c10::irange(sm))
  {
    nnz += kkijs[i].size();
  }
  // std::cout << "nnz: " << nnz << std::endl;
  // flatten kkijs
  index_t *data_ptr = (index_t *)malloc(nnz * 5 * sizeof(index_t));
  index_t *unique_ijs = (index_t *)malloc(nnz * 2 * sizeof(index_t));
  std::unordered_map<index_t, index_t> unique_ij_map;

  int cur = 0;
  for (const auto i : c10::irange(sm))
  {
    for (const auto &kkij : kkijs[i])
    {
      auto k1 = std::get<0>(kkij);
      auto k2 = std::get<1>(kkij);
      auto i = std::get<2>(kkij);
      auto j = std::get<3>(kkij);
      auto ij = i * sm + j;
      if (auto search = unique_ij_map.find(ij); search != unique_ij_map.end())
      {
        data_ptr[cur * 5 + 4] = search->second;
      }
      else
      {
        unique_ijs[2*nuijs] = i;
        unique_ijs[2*nuijs+1] = j;
        data_ptr[cur * 5 + 4] = nuijs;
        unique_ij_map[ij] = nuijs;
        nuijs += 1;
      }

      data_ptr[cur * 5] = k1;
      data_ptr[cur * 5 + 1] = k2;
      data_ptr[cur * 5 + 2] = i;
      data_ptr[cur * 5 + 3] = j;
      cur += 1;
    }
  }
  // torch::Tensor result = torch::from_blob(data_ptr, {nnz, 5});
  // torch::Tensor coo_indices = torch::from_blob(unique_ijs.data(), {unique_ijs.size(), 2});
  return std::make_tuple(data_ptr, nnz, unique_ijs, nuijs);
}

int called_count = 0;
torch::Tensor sources;
torch::Tensor coo_indices;

torch::Tensor sparse_bsr_mm(const torch::Tensor &bsr, const torch::Tensor &bsc)
{
  auto crow_indices = bsr.crow_indices().cpu().contiguous();
  auto col_indices = bsr.col_indices().cpu().contiguous();
  auto csr_values = bsr.values();

  auto ccol_indices = bsc.ccol_indices().cpu().contiguous();
  auto row_indices = bsc.row_indices().cpu().contiguous();
  auto csc_values = bsc.values();

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


  int nnz;
  int nuijs;
  if (called_count == 0) {AT_DISPATCH_INDEX_TYPES(
      crow_indices.scalar_type(),
      "bsr_mm_crow_indices",
      [&]()
      {
        auto symbols = scan_symbol<index_t>(crow_indices.data_ptr<index_t>(),
                                            ccol_indices.data_ptr<index_t>(),
                                            row_indices.data_ptr<index_t>(),
                                            col_indices.data_ptr<index_t>(),
                                            sm, sp);
        auto data_ptr = std::get<0>(symbols);
        auto unique_ijs = std::get<2>(symbols);
        // for (const auto i : c10::irange(std::get<1>(symbols)))
        // {
        //   std::cout << data_ptr[i * 5] << " " << data_ptr[i * 5 + 1] << " " << data_ptr[i * 5 + 2] << " " << data_ptr[i * 5 + 3] << " " << data_ptr[i * 5 + 4] << std::endl;
        // }
        nnz = std::get<1>(symbols);
        nuijs = std::get<3>(symbols);
        sources = torch::from_blob(data_ptr, {nnz, 5}, crow_indices.options()).to(csr_values.device());
        // std::cout << "index: " << index << std::endl;
        //print unique_ijs
        // for (const auto i : c10::irange(nuijs))
        // {
        //   std::cout << unique_ijs[i * 2] << " " << unique_ijs[i * 2 + 1] << std::endl;
        // }
        coo_indices = torch::from_blob(unique_ijs, {nuijs, 2}, crow_indices.options()).to(csr_values.device());
      });}
  // std::cout << "nuijs: " << nuijs << std::endl; 
  // std::cout << "coo_indices[..., 0]: " << coo_indices.index({"...", 0}) << std::endl;
  // print coo_indices using tensor accessor
  // for (const auto i : c10::irange(nuijs)) {
  //   std::cout << coo_indices[i][0].item<int>() << " " << coo_indices[i][1].item<int>() << std::endl;
  // }
  auto index = sources.index({"...", 4});
  auto prod = torch::bmm(csr_values.index({sources.index({"...", 0})}), csc_values.index({sources.index({"...", 1})}));
  auto reduced = torch::zeros({nuijs, dm, dp}, prod.options());
  reduced.scatter_add_(0, index.index({"...", torch::indexing::None, torch::indexing::None}).expand_as(prod), prod);
  auto row_res = coo_indices.index({"...", 0});
  auto col_res = coo_indices.index({"...", 1});
  // std::cout << "sm, sp: " << sm << " " << sp << std::endl;
  // std::cout << "row_res: " << row_res << std::endl;
  // std::cout << "col_res: " << col_res << std::endl;
  auto crow_res = at::_convert_indices_from_coo_to_csr(row_res.contiguous(), sm, row_res.dtype() == at::kInt);
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
  called_count += 1;
  return at::_sparse_compressed_tensor_unsafe(crow_res.to(reduced.device()), col_res.to(reduced.device()), reduced.to(reduced.device()), {m, p}, at::TensorOptions().dtype(reduced.dtype()).device(reduced.device()).layout(at::kSparseBsr));
}

torch::Tensor sparse_bsr_csr_mm(const torch::Tensor &a, const torch::Tensor &b)
{
  if (a.layout() == at::kSparseBsr && b.layout() == at::kSparseBsc)
  {
    // auto output = (*_sparse_bsr_bsc_matmul)({a, b});
    // return output.toTensor();
    return sparse_bsr_mm(a, b);
  }
# if AT_USE_MKL_SPARSE()
  return at::native::_sparse_csr_mm(a, b);
# else
  // TORCH_CHECK(false, "MKL Sparse is not enabled.");
  if (a.layout() == at::kSparseBsr && b.layout() == at::kStrided)
  {
    int64_t b_shape[2] = {a.values().size(-1), b.size(-1)};
    torch::Tensor bsc = b.to_sparse_bsc(b_shape);
    // std::cout << "b_shape: " << b_shape[0] << " " << b_shape[1] << std::endl;
    return sparse_bsr_mm(a, bsc).to_dense();
  }
# endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparse_bsr_mm", &sparse_bsr_mm, "Sparse BSR CSR matrix multiplication");
}
TORCH_LIBRARY_IMPL(aten, SparseCsrCPU, m)
{
  // hide stderr
  // freopen("/dev/null", "a", stderr);
  // auto module = pybind11::module::import("pypose.sparse.ops");
  // auto object = module.attr("bsr_bsc_matmul");
  // auto script_function = object.cast<torch::jit::StrongFunctionPtr>();
  // _sparse_bsr_bsc_matmul = script_function.function_;
  // m.impl("mm", sparse_bsr_csr_mm);
  // resume stderr
  // freopen("/dev/tty", "a", stderr);
}


/*
the file that calls the cuBLAS API for matrix multiplication is located at aten/src/ATen/native/cuda/Blas.cpp. In this file, there is a function called matmul_kernel that dispatches different cuBLAS functions based on the input tensor shapes and data types. For example, if the input tensors are 2-dimensional and have floating-point values, the function will call cublasSgemm or cublasDgemm for single-precision or double-precision arithmetic, respectively.

the file that calls the MKL API for matrix multiplication is located at aten/src/ATen/native/LinearAlgebra.cpp. In this file, there is a function called bmm_out_or_baddbmm_ that dispatches different MKL functions based on the input tensor shapes and data types. For example, if the input tensors are 2-dimensional and have floating-point values, the function will call cblas_sgemm or cblas_dgemm for single-precision or double-precision arithmetic, respectively.

Blas: same file but in different function
at::blas::gemm
*/