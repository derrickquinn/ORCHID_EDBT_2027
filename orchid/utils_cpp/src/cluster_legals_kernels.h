#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

static inline uint32_t popcount_bytes(const uint8_t* data, int32_t len) {
  uint64_t total = 0;
  int32_t i = 0;
  for (; i + 8 <= len; i += 8) {
    uint64_t v;
    std::memcpy(&v, data + i, sizeof(uint64_t));
    total += static_cast<uint64_t>(__builtin_popcountll(v));
  }
  for (; i < len; ++i) {
    total += static_cast<uint64_t>(__builtin_popcount(data[i]));
  }
  return static_cast<uint32_t>(total);
}

static inline int32_t resolve_threads(int32_t threads) {
#ifdef _OPENMP
  if (threads > 0) {
    return threads;
  }
  return omp_get_max_threads();
#else
  (void)threads;
  return 1;
#endif
}

static void ids_to_bitmap_impl(
    const int64_t* offsets,
    const uint32_t* ids,
    int32_t n_queries,
    int32_t bytes_per_query,
    uint8_t* out,
    bool clear,
    int32_t threads) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
  for (int32_t q = 0; q < n_queries; ++q) {
    uint8_t* row = out + static_cast<int64_t>(q) * bytes_per_query;
    if (clear) {
      std::memset(row, 0, static_cast<size_t>(bytes_per_query));
    }
    for (int64_t i = offsets[q]; i < offsets[q + 1]; ++i) {
      const uint32_t id = ids[i];
      row[id >> 3] |= static_cast<uint8_t>(1u << (id & 7u));
    }
  }
}

static void predicate_results_to_bitmap_impl(
    const uint32_t* const* posting_lists,
    const int64_t* posting_sizes,
    const uint8_t* const* dense_bitmaps,
    int32_t n_queries,
    int32_t bytes_per_query,
    uint8_t* out,
    int32_t threads) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
  for (int32_t q = 0; q < n_queries; ++q) {
    uint8_t* row = out + static_cast<int64_t>(q) * bytes_per_query;
    if (dense_bitmaps[q] != nullptr) {
      std::memcpy(row, dense_bitmaps[q], static_cast<size_t>(bytes_per_query));
      continue;
    }
    std::memset(row, 0, static_cast<size_t>(bytes_per_query));
    for (int64_t i = 0; i < posting_sizes[q]; ++i) {
      const uint32_t id = posting_lists[q][i];
      row[id >> 3] |= static_cast<uint8_t>(1u << (id & 7u));
    }
  }
}

static void predicate_results_to_mask_impl(const uint32_t* const* posting_lists,
                                           const int64_t* posting_sizes,
                                           const uint8_t* const* dense_masks,
                                           int32_t n_queries,
                                           int32_t n_docs,
                                           uint8_t* out,
                                           int32_t threads) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
  for (int32_t q = 0; q < n_queries; ++q) {
    uint8_t* row = out + static_cast<int64_t>(q) * n_docs;
    if (dense_masks[q] != nullptr) {
      std::memcpy(row, dense_masks[q], static_cast<size_t>(n_docs));
      continue;
    }
    std::memset(row, 0, static_cast<size_t>(n_docs));
    for (int64_t i = 0; i < posting_sizes[q]; ++i) {
      row[posting_lists[q][i]] = 1;
    }
  }
}

static void cluster_legals_preordered_packed_impl(
    const uint8_t* mask_packed,
    int32_t n_queries,
    int32_t n_docs,
    int32_t bytes_per_query,
    const int32_t* cluster_offsets,
    const int32_t* cluster_counts,
    int32_t n_list,
    float* out,
    int32_t threads) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
  for (int32_t q = 0; q < n_queries; ++q) {
    const uint8_t* row =
        mask_packed + static_cast<int64_t>(q) * bytes_per_query;
    float* out_row = out + static_cast<int64_t>(q) * n_list;
    for (int32_t c = 0; c < n_list; ++c) {
      int32_t start = cluster_offsets[c];
      int32_t end = cluster_offsets[c + 1];
      if (start >= end) {
        out_row[c] = 0.0f;
        continue;
      }

      int32_t s_byte = start >> 3;
      int32_t e_byte = end >> 3;
      int32_t s_bit = start & 7;
      int32_t e_bit = end & 7;

      uint32_t sum = 0;
      if (s_byte == e_byte) {
        uint8_t mask =
            static_cast<uint8_t>(((1u << (e_bit - s_bit)) - 1u) << s_bit);
        sum += static_cast<uint32_t>(__builtin_popcount(row[s_byte] & mask));
      } else {
        if (s_bit != 0) {
          uint8_t mask = static_cast<uint8_t>(0xFFu << s_bit);
          sum += static_cast<uint32_t>(__builtin_popcount(row[s_byte] & mask));
          s_byte += 1;
        }

        if (s_byte < e_byte) {
          sum += popcount_bytes(row + s_byte, e_byte - s_byte);
        }

        if (e_bit != 0) {
          uint8_t mask = static_cast<uint8_t>((1u << e_bit) - 1u);
          sum += static_cast<uint32_t>(__builtin_popcount(row[e_byte] & mask));
        }
      }

      int32_t cnt = cluster_counts[c];
      out_row[c] = cnt > 0 ? static_cast<float>(sum) / static_cast<float>(cnt)
                           : 0.0f;
    }
  }
}

static void log_scale_impl(
    const float* in,
    int32_t n_queries,
    int32_t n_list,
    float beta,
    float eps,
    float* out,
    int32_t threads) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
  for (int32_t q = 0; q < n_queries; ++q) {
    const float* in_row = in + static_cast<int64_t>(q) * n_list;
    float* out_row = out + static_cast<int64_t>(q) * n_list;
#ifdef _OPENMP
#pragma omp simd
#endif
    for (int32_t c = 0; c < n_list; ++c) {
      out_row[c] = beta * std::log(in_row[c] + eps);
    }
  }
}
