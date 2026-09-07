#include <algorithm>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "cluster_legals_kernels.h"
#include "predicate_eval_kernels.h"
#include "topk_kernels.h"

namespace py = pybind11;

static py::array_t<int64_t> topk_small_out_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> scores,
    int32_t k,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> out,
    int32_t threads) {
  if (scores.ndim() != 2) {
    throw std::runtime_error("scores must be 2D [rows, width]");
  }
  const int32_t rows = static_cast<int32_t>(scores.shape(0));
  const int32_t width = static_cast<int32_t>(scores.shape(1));
  if (k <= 0 || k > width) {
    throw std::runtime_error("k must be in [1, width]");
  }
  if (out.ndim() != 2 || out.shape(0) != static_cast<size_t>(rows) ||
      out.shape(1) != static_cast<size_t>(k)) {
    throw std::runtime_error("out shape must be (rows, k)");
  }

  const int32_t threads_resolved = resolve_threads(threads);
  {
    py::gil_scoped_release release;
    topk_small_impl(scores.data(), rows, width, k, out.mutable_data(),
                    threads_resolved);
  }
  return out;
}

static py::array_t<uint8_t> ids_to_bitmap_out_py(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> offsets,
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> ids,
    int32_t n_docs,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> out,
    bool clear,
    int32_t threads) {
  if (offsets.ndim() != 1 || offsets.shape(0) < 1) {
    throw std::runtime_error("offsets must be 1D [n_queries+1]");
  }
  if (ids.ndim() != 1) {
    throw std::runtime_error("ids must be 1D");
  }
  if (out.ndim() != 2) {
    throw std::runtime_error("out must be 2D [n_queries, n_bytes]");
  }
  if (n_docs <= 0) {
    throw std::runtime_error("n_docs must be positive");
  }

  const int32_t n_queries = static_cast<int32_t>(offsets.shape(0) - 1);
  const int32_t bytes_per_query = (n_docs + 7) >> 3;
  if (out.shape(0) != static_cast<size_t>(n_queries) ||
      out.shape(1) != static_cast<size_t>(bytes_per_query)) {
    throw std::runtime_error(
        "out shape must be (n_queries, ceil(n_docs / 8))");
  }

  const int64_t* offsets_ptr = offsets.data();
  if (offsets_ptr[0] != 0 ||
      offsets_ptr[n_queries] != static_cast<int64_t>(ids.shape(0))) {
    throw std::runtime_error("offsets must span the complete ids array");
  }
  for (int32_t q = 0; q < n_queries; ++q) {
    if (offsets_ptr[q] > offsets_ptr[q + 1]) {
      throw std::runtime_error("offsets must be nondecreasing");
    }
  }

  const uint32_t* ids_ptr = ids.data();
  uint8_t* out_ptr = out.mutable_data();
  const int32_t threads_resolved = resolve_threads(threads);
  {
    // Callers may range-check when a result set is prepared. Keeping that
    // scan off this hot path is intentional.
    py::gil_scoped_release release;
    ids_to_bitmap_impl(offsets_ptr, ids_ptr, n_queries, bytes_per_query,
                       out_ptr, clear, threads_resolved);
  }
  return out;
}

static py::array_t<uint8_t> predicate_results_to_bitmap_out_py(
    py::sequence posting_lists,
    py::sequence dense_bitmaps,
    int32_t n_docs,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> out,
    int32_t threads) {
  if (out.ndim() != 2 || n_docs <= 0) {
    throw std::runtime_error("invalid predicate bitmap inputs");
  }
  const int32_t n_queries = static_cast<int32_t>(posting_lists.size());
  const int32_t bytes_per_query = (n_docs + 7) >> 3;
  if (static_cast<int32_t>(dense_bitmaps.size()) != n_queries) {
    throw std::runtime_error("dense_bitmaps must have one entry per query");
  }
  if (out.shape(0) != static_cast<size_t>(n_queries) ||
      out.shape(1) != static_cast<size_t>(bytes_per_query)) {
    throw std::runtime_error(
        "out shape must be (n_queries, ceil(n_docs / 8))");
  }
  using PostingArray =
      py::array_t<uint32_t, py::array::c_style | py::array::forcecast>;
  using DenseArray =
      py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
  std::vector<PostingArray> posting_arrays;
  std::vector<DenseArray> dense_arrays;
  std::vector<const uint32_t*> posting_ptrs(static_cast<size_t>(n_queries),
                                             nullptr);
  std::vector<int64_t> posting_sizes(static_cast<size_t>(n_queries), 0);
  std::vector<const uint8_t*> dense_ptrs(static_cast<size_t>(n_queries),
                                          nullptr);
  posting_arrays.reserve(static_cast<size_t>(n_queries));
  dense_arrays.reserve(static_cast<size_t>(n_queries));
  for (int32_t q = 0; q < n_queries; ++q) {
    py::handle posting_item = posting_lists[q];
    py::handle dense_item = dense_bitmaps[q];
    if (posting_item.is_none() == dense_item.is_none()) {
      throw std::runtime_error(
          "each query must have either postings or a dense bitmap");
    }
    if (!posting_item.is_none()) {
      PostingArray postings = PostingArray::ensure(posting_item);
      if (!postings || postings.ndim() != 1) {
        throw std::runtime_error("posting result must be one-dimensional");
      }
      posting_arrays.push_back(std::move(postings));
      posting_ptrs[q] = posting_arrays.back().data();
      posting_sizes[q] = posting_arrays.back().size();
      continue;
    }
    DenseArray bitmap = DenseArray::ensure(dense_item);
    if (!bitmap || bitmap.ndim() != 1 ||
        bitmap.shape(0) != static_cast<size_t>(bytes_per_query)) {
      throw std::runtime_error("dense bitmap has the wrong width");
    }
    dense_arrays.push_back(std::move(bitmap));
    dense_ptrs[q] = dense_arrays.back().data();
  }

  const int32_t threads_resolved = resolve_threads(threads);
  {
    py::gil_scoped_release release;
    predicate_results_to_bitmap_impl(
        posting_ptrs.data(), posting_sizes.data(), dense_ptrs.data(), n_queries,
        bytes_per_query, out.mutable_data(), threads_resolved);
  }
  return out;
}

static py::array_t<uint8_t> predicate_results_to_mask_out_py(
    py::sequence posting_lists,
    py::sequence dense_masks,
    int32_t n_docs,
    py::array_t<uint8_t, py::array::c_style> out,
    int32_t threads) {
  if (out.ndim() != 2 || n_docs <= 0) {
    throw std::runtime_error("invalid predicate mask inputs");
  }
  const int32_t n_queries = static_cast<int32_t>(posting_lists.size());
  if (static_cast<int32_t>(dense_masks.size()) != n_queries) {
    throw std::runtime_error("dense_masks must have one entry per query");
  }
  if (out.shape(0) != static_cast<size_t>(n_queries) ||
      out.shape(1) != static_cast<size_t>(n_docs)) {
    throw std::runtime_error("out shape must be (n_queries, n_docs)");
  }
  using PostingArray =
      py::array_t<uint32_t, py::array::c_style | py::array::forcecast>;
  using DenseArray =
      py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
  std::vector<PostingArray> posting_arrays;
  std::vector<DenseArray> dense_arrays;
  std::vector<const uint32_t*> posting_ptrs(static_cast<size_t>(n_queries),
                                            nullptr);
  std::vector<int64_t> posting_sizes(static_cast<size_t>(n_queries), 0);
  std::vector<const uint8_t*> dense_ptrs(static_cast<size_t>(n_queries),
                                         nullptr);
  posting_arrays.reserve(static_cast<size_t>(n_queries));
  dense_arrays.reserve(static_cast<size_t>(n_queries));
  for (int32_t q = 0; q < n_queries; ++q) {
    py::handle posting_item = posting_lists[q];
    py::handle dense_item = dense_masks[q];
    if (posting_item.is_none() == dense_item.is_none()) {
      throw std::runtime_error(
          "each query must have either postings or a dense bitmap");
    }
    if (!posting_item.is_none()) {
      PostingArray postings = PostingArray::ensure(posting_item);
      if (!postings || postings.ndim() != 1) {
        throw std::runtime_error("posting result must be one-dimensional");
      }
      posting_arrays.push_back(std::move(postings));
      posting_ptrs[q] = posting_arrays.back().data();
      posting_sizes[q] = posting_arrays.back().size();
      continue;
    }
    DenseArray bitmap = DenseArray::ensure(dense_item);
    if (!bitmap || bitmap.ndim() != 1 ||
        bitmap.shape(0) != static_cast<size_t>(n_docs)) {
      throw std::runtime_error("dense byte mask has the wrong width");
    }
    dense_arrays.push_back(std::move(bitmap));
    dense_ptrs[q] = dense_arrays.back().data();
  }

  const int32_t threads_resolved = resolve_threads(threads);
  {
    py::gil_scoped_release release;
    predicate_results_to_mask_impl(posting_ptrs.data(),
                                   posting_sizes.data(),
                                   dense_ptrs.data(),
                                   n_queries,
                                   n_docs,
                                   out.mutable_data(),
                                   threads_resolved);
  }
  return out;
}

static py::array_t<float> cluster_legals_preordered_packed_out_py(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> mask_packed,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast>
        cluster_offsets,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast>
        cluster_counts,
    py::array_t<float, py::array::c_style | py::array::forcecast> out,
    int32_t threads) {
  if (mask_packed.ndim() != 2) {
    throw std::runtime_error("mask_packed must be 2D [n_queries, n_bytes]");
  }
  if (cluster_offsets.ndim() != 1) {
    throw std::runtime_error("cluster_offsets must be 1D [n_list+1]");
  }
  if (cluster_counts.ndim() != 1) {
    throw std::runtime_error("cluster_counts must be 1D [n_list]");
  }
  if (out.ndim() != 2) {
    throw std::runtime_error("out must be 2D [n_queries, n_list]");
  }

  int32_t n_queries = static_cast<int32_t>(mask_packed.shape(0));
  int32_t n_list = static_cast<int32_t>(cluster_counts.shape(0));
  if (cluster_offsets.shape(0) != static_cast<size_t>(n_list + 1)) {
    throw std::runtime_error("cluster_offsets length must be n_list + 1");
  }
  if (out.shape(0) != static_cast<size_t>(n_queries) ||
      out.shape(1) != static_cast<size_t>(n_list)) {
    throw std::runtime_error("out shape must be (n_queries, n_list)");
  }

  int32_t n_docs = cluster_offsets.data()[n_list];
  int32_t bytes_per_query = static_cast<int32_t>(mask_packed.shape(1));
  int32_t expected_bytes = (n_docs + 7) >> 3;
  if (bytes_per_query != expected_bytes) {
    throw std::runtime_error("mask_packed shape does not match cluster_offsets");
  }

  int32_t threads_resolved = resolve_threads(threads);
  const uint8_t* mask_ptr = mask_packed.data();
  const int32_t* offsets_ptr = cluster_offsets.data();
  const int32_t* counts_ptr = cluster_counts.data();
  float* out_ptr = out.mutable_data();

  {
    py::gil_scoped_release release;
    cluster_legals_preordered_packed_impl(mask_ptr, n_queries, n_docs,
                                          bytes_per_query, offsets_ptr,
                                          counts_ptr, n_list, out_ptr,
                                          threads_resolved);
  }

  return out;
}

static py::array_t<float> log_scale_out_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> in,
    float beta,
    py::array_t<float, py::array::c_style | py::array::forcecast> out,
    float eps,
    int32_t threads) {
  if (in.ndim() != 2) {
    throw std::runtime_error("in must be 2D [n_queries, n_list]");
  }
  if (out.ndim() != 2) {
    throw std::runtime_error("out must be 2D [n_queries, n_list]");
  }
  if (in.shape(0) != out.shape(0) || in.shape(1) != out.shape(1)) {
    throw std::runtime_error("out shape must match in shape");
  }

  int32_t n_queries = static_cast<int32_t>(in.shape(0));
  int32_t n_list = static_cast<int32_t>(in.shape(1));

  int32_t threads_resolved = resolve_threads(threads);
  const float* in_ptr = in.data();
  float* out_ptr = out.mutable_data();

  {
    py::gil_scoped_release release;
    log_scale_impl(in_ptr, n_queries, n_list, beta, eps, out_ptr,
                   threads_resolved);
  }

  return out;
}

static py::tuple prepare_cluster_order_py(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast>
        doc_cluster,
    int32_t n_list) {
  if (doc_cluster.ndim() != 1) {
    throw std::runtime_error("doc_cluster must be 1D [n_docs]");
  }

  int32_t n_docs = static_cast<int32_t>(doc_cluster.shape(0));
  const int32_t* cluster_ptr = doc_cluster.data();

  if (n_list <= 0) {
    throw std::runtime_error("n_list must be positive");
  }

  py::array_t<int32_t> cluster_counts({n_list});
  py::array_t<int32_t> cluster_offsets({n_list + 1});
  py::array_t<int32_t> perm({n_docs});

  int32_t* counts_ptr = cluster_counts.mutable_data();
  std::fill(counts_ptr, counts_ptr + n_list, 0);

  for (int32_t d = 0; d < n_docs; ++d) {
    int32_t c = cluster_ptr[d];
    if (c < 0 || c >= n_list) {
      throw std::runtime_error("doc_cluster contains out-of-range id");
    }
    counts_ptr[c] += 1;
  }

  int32_t* offsets_ptr = cluster_offsets.mutable_data();
  offsets_ptr[0] = 0;
  for (int32_t c = 0; c < n_list; ++c) {
    offsets_ptr[c + 1] = offsets_ptr[c] + counts_ptr[c];
  }

  std::vector<int32_t> cursor(n_list);
  std::copy(offsets_ptr, offsets_ptr + n_list, cursor.begin());
  int32_t* perm_ptr = perm.mutable_data();
  for (int32_t d = 0; d < n_docs; ++d) {
    int32_t c = cluster_ptr[d];
    perm_ptr[cursor[c]++] = d;
  }

  return py::make_tuple(perm, cluster_offsets, cluster_counts);
}

static py::array_t<uint8_t> reorder_mask_py(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> mask,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> perm) {
  if (mask.ndim() != 2) {
    throw std::runtime_error("mask must be 2D [n_queries, n_docs]");
  }
  if (perm.ndim() != 1) {
    throw std::runtime_error("perm must be 1D [n_docs]");
  }

  int32_t n_queries = static_cast<int32_t>(mask.shape(0));
  int32_t n_docs = static_cast<int32_t>(mask.shape(1));
  if (perm.shape(0) != static_cast<size_t>(n_docs)) {
    throw std::runtime_error("perm length must match n_docs");
  }

  auto out = py::array_t<uint8_t>({n_queries, n_docs});
  const uint8_t* mask_ptr = mask.data();
  const int32_t* perm_ptr = perm.data();
  uint8_t* out_ptr = out.mutable_data();

  {
    py::gil_scoped_release release;
    for (int32_t q = 0; q < n_queries; ++q) {
      const uint8_t* src = mask_ptr + static_cast<int64_t>(q) * n_docs;
      uint8_t* dst = out_ptr + static_cast<int64_t>(q) * n_docs;
      for (int32_t i = 0; i < n_docs; ++i) {
        dst[i] = src[perm_ptr[i]];
      }
    }
  }

  return out;
}

static py::tuple evaluate_predicate_conjunctions_py(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> keys,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> counts,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast>
        sparse_offsets,
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> postings,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> dense_rows,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> dense_masks,
    int32_t n_docs,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast>
        query_offsets,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> query_terms,
    bool packed,
    int32_t threads) {
  if (keys.ndim() != 1 || counts.ndim() != 1 || sparse_offsets.ndim() != 1 ||
      postings.ndim() != 1 || dense_rows.ndim() != 1) {
    throw std::runtime_error("predicate-index vectors must be one-dimensional");
  }
  if (dense_masks.ndim() != 2) {
    throw std::runtime_error("dense_masks must be two-dimensional");
  }
  if (query_offsets.ndim() != 1 || query_offsets.shape(0) < 1 ||
      query_terms.ndim() != 1) {
    throw std::runtime_error("queries must be CSR offsets and terms");
  }
  if (n_docs <= 0) {
    throw std::runtime_error("n_docs must be positive");
  }

  const int32_t n_keys = static_cast<int32_t>(keys.shape(0));
  if (counts.shape(0) != static_cast<size_t>(n_keys) ||
      sparse_offsets.shape(0) != static_cast<size_t>(n_keys + 1) ||
      dense_rows.shape(0) != static_cast<size_t>(n_keys)) {
    throw std::runtime_error("predicate-index shapes disagree");
  }
  if (sparse_offsets.data()[0] != 0 ||
      sparse_offsets.data()[n_keys] != static_cast<int64_t>(postings.shape(0))) {
    throw std::runtime_error("sparse_offsets must span postings");
  }
  const int32_t dense_width = packed ? (n_docs + 7) >> 3 : n_docs;
  if (dense_masks.shape(1) != static_cast<size_t>(dense_width)) {
    throw std::runtime_error("dense-mask width does not match n_docs");
  }

  const int32_t n_queries =
      static_cast<int32_t>(query_offsets.shape(0) - 1);
  const int64_t* query_offsets_ptr = query_offsets.data();
  if (query_offsets_ptr[0] != 0 ||
      query_offsets_ptr[n_queries] !=
          static_cast<int64_t>(query_terms.shape(0))) {
    throw std::runtime_error("query_offsets must span query_terms");
  }
  for (int32_t query = 0; query < n_queries; ++query) {
    if (query_offsets_ptr[query] >= query_offsets_ptr[query + 1]) {
      throw std::runtime_error(
          "each native conjunction must contain at least one term");
    }
  }

  std::vector<PredicateQueryResult> results(static_cast<size_t>(n_queries));
  const int32_t threads_resolved = resolve_threads(threads);
  {
    py::gil_scoped_release release;
    if (packed) {
      evaluate_predicate_conjunctions_impl<true>(
          keys.data(), counts.data(), sparse_offsets.data(), postings.data(),
          dense_rows.data(), dense_masks.data(), n_keys, dense_width,
          query_offsets_ptr, query_terms.data(), n_queries, results,
          threads_resolved);
    } else {
      evaluate_predicate_conjunctions_impl<false>(
          keys.data(), counts.data(), sparse_offsets.data(), postings.data(),
          dense_rows.data(), dense_masks.data(), n_keys, dense_width,
          query_offsets_ptr, query_terms.data(), n_queries, results,
          threads_resolved);
    }
  }

  int64_t sparse_size = 0;
  int32_t dense_count = 0;
  for (const PredicateQueryResult& result : results) {
    if (result.dense) {
      ++dense_count;
    } else {
      sparse_size += static_cast<int64_t>(result.ids.size());
    }
  }

  py::array_t<uint8_t> kinds({n_queries});
  py::array_t<int64_t> result_offsets({n_queries + 1});
  py::array_t<uint32_t> result_ids({sparse_size});
  py::array_t<int32_t> result_dense_rows({n_queries});
  py::array_t<uint8_t> result_dense_masks({dense_count, dense_width});
  uint8_t* kinds_ptr = kinds.mutable_data();
  int64_t* offsets_ptr = result_offsets.mutable_data();
  uint32_t* ids_ptr = result_ids.mutable_data();
  int32_t* dense_rows_ptr = result_dense_rows.mutable_data();
  uint8_t* dense_masks_ptr = result_dense_masks.mutable_data();
  offsets_ptr[0] = 0;
  int64_t id_cursor = 0;
  int32_t dense_cursor = 0;
  for (int32_t query = 0; query < n_queries; ++query) {
    const PredicateQueryResult& result = results[query];
    if (result.dense) {
      kinds_ptr[query] = 1;
      dense_rows_ptr[query] = dense_cursor;
      std::memcpy(dense_masks_ptr + static_cast<int64_t>(dense_cursor) *
                                        dense_width,
                  result.dense_mask.data(), static_cast<size_t>(dense_width));
      ++dense_cursor;
    } else {
      kinds_ptr[query] = 0;
      dense_rows_ptr[query] = -1;
      std::copy(result.ids.begin(), result.ids.end(), ids_ptr + id_cursor);
      id_cursor += static_cast<int64_t>(result.ids.size());
    }
    offsets_ptr[query + 1] = id_cursor;
  }
  if (offsets_ptr[n_queries] != sparse_size || id_cursor != sparse_size ||
      dense_cursor != dense_count) {
    throw std::runtime_error("predicate result assembly failed");
  }

  return py::make_tuple(kinds, result_offsets, result_ids, result_dense_rows,
                        result_dense_masks);
}

PYBIND11_MODULE(_cluster_legals, m) {
  m.doc() = "Cluster legals kernel (preordered packed)";

  m.def("prepare_cluster_order", &prepare_cluster_order_py,
        py::arg("doc_cluster"), py::arg("n_list"));
  m.def("reorder_mask", &reorder_mask_py, py::arg("mask"), py::arg("perm"));
  m.def("topk_small_out", &topk_small_out_py, py::arg("scores"),
        py::arg("k"), py::arg("out"), py::arg("threads") = 0);
  m.def("ids_to_bitmap_out", &ids_to_bitmap_out_py, py::arg("offsets"),
        py::arg("ids"), py::arg("n_docs"), py::arg("out"),
        py::arg("clear") = true, py::arg("threads") = 0);
  m.def("cluster_legals_preordered_packed_out",
        &cluster_legals_preordered_packed_out_py, py::arg("mask_packed"),
        py::arg("cluster_offsets"), py::arg("cluster_counts"), py::arg("out"),
        py::arg("threads") = 0);
  m.def("log_scale_out", &log_scale_out_py, py::arg("in"), py::arg("beta"),
        py::arg("out"), py::arg("eps") = 1e-9f, py::arg("threads") = 0);
  m.def("predicate_results_to_bitmap_out",
        &predicate_results_to_bitmap_out_py, py::arg("posting_lists"),
        py::arg("dense_bitmaps"), py::arg("n_docs"), py::arg("out"),
        py::arg("threads") = 0);
  m.def("predicate_results_to_mask_out",
        &predicate_results_to_mask_out_py, py::arg("posting_lists"),
        py::arg("dense_masks"), py::arg("n_docs"),
        py::arg("out").noconvert(),
        py::arg("threads") = 0);
  m.def("evaluate_predicate_conjunctions",
        &evaluate_predicate_conjunctions_py, py::arg("keys"),
        py::arg("counts"), py::arg("sparse_offsets"), py::arg("postings"),
        py::arg("dense_rows"), py::arg("dense_masks"), py::arg("n_docs"),
        py::arg("query_offsets"), py::arg("query_terms"),
        py::arg("packed") = true,
        py::arg("threads") = 0);
}
