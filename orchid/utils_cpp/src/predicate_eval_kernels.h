#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

struct PredicateQueryResult {
  bool dense = false;
  std::vector<uint32_t> ids;
  std::vector<uint8_t> dense_mask;
};

static inline int32_t find_predicate_key(const int64_t* keys, int32_t n_keys,
                                         int64_t term) {
  const int64_t* found = std::lower_bound(keys, keys + n_keys, term);
  if (found == keys + n_keys || *found != term) {
    return -1;
  }
  return static_cast<int32_t>(found - keys);
}

static inline std::vector<uint32_t> intersect_sorted_postings(
    const std::vector<uint32_t>& left, const uint32_t* right,
    int64_t right_size) {
  std::vector<uint32_t> result;
  result.reserve(std::min<int64_t>(left.size(), right_size));
  size_t left_pos = 0;
  int64_t right_pos = 0;
  while (left_pos < left.size() && right_pos < right_size) {
    const uint32_t left_id = left[left_pos];
    const uint32_t right_id = right[right_pos];
    if (left_id < right_id) {
      ++left_pos;
    } else if (right_id < left_id) {
      ++right_pos;
    } else {
      result.push_back(left_id);
      ++left_pos;
      ++right_pos;
    }
  }
  return result;
}

template <bool Packed>
static inline void filter_postings_by_dense_mask(std::vector<uint32_t>& ids,
                                                 const uint8_t* mask) {
  size_t out = 0;
  for (const uint32_t id : ids) {
    const bool present = Packed
                             ? (mask[id >> 3] & static_cast<uint8_t>(
                                                    1u << (id & 7u))) != 0
                             : mask[id] != 0;
    if (present) {
      ids[out++] = id;
    }
  }
  ids.resize(out);
}

template <bool Packed>
static void evaluate_predicate_conjunctions_impl(
    const int64_t* keys, const int64_t* counts, const int64_t* sparse_offsets,
    const uint32_t* postings, const int32_t* dense_rows,
    const uint8_t* dense_masks, int32_t n_keys, int32_t dense_width,
    const int64_t* query_offsets, const int64_t* query_terms,
    int32_t n_queries, std::vector<PredicateQueryResult>& results,
    int32_t threads) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
#endif
  for (int32_t query = 0; query < n_queries; ++query) {
    std::vector<int32_t> key_indices;
    key_indices.reserve(
        static_cast<size_t>(query_offsets[query + 1] - query_offsets[query]));
    bool missing = false;
    for (int64_t term_pos = query_offsets[query];
         term_pos < query_offsets[query + 1]; ++term_pos) {
      const int32_t key =
          find_predicate_key(keys, n_keys, query_terms[term_pos]);
      if (key < 0) {
        missing = true;
        break;
      }
      key_indices.push_back(key);
    }
    if (missing) {
      continue;
    }

    std::sort(key_indices.begin(), key_indices.end(),
              [counts](int32_t left, int32_t right) {
                return counts[left] < counts[right];
              });
    key_indices.erase(
        std::unique(key_indices.begin(), key_indices.end()),
        key_indices.end());

    PredicateQueryResult& result = results[query];
    const bool all_dense =
        std::all_of(key_indices.begin(), key_indices.end(),
                    [dense_rows](int32_t key) { return dense_rows[key] >= 0; });
    if (all_dense) {
      result.dense = true;
      result.dense_mask.resize(static_cast<size_t>(dense_width));
      const uint8_t* first =
          dense_masks + static_cast<int64_t>(dense_rows[key_indices[0]]) *
                            dense_width;
      std::memcpy(result.dense_mask.data(), first,
                  static_cast<size_t>(dense_width));
      for (size_t operand = 1; operand < key_indices.size(); ++operand) {
        const uint8_t* next =
            dense_masks +
            static_cast<int64_t>(dense_rows[key_indices[operand]]) *
                dense_width;
        uint8_t* out = result.dense_mask.data();
#ifdef _OPENMP
#pragma omp simd
#endif
        for (int32_t position = 0; position < dense_width; ++position) {
          out[position] &= next[position];
        }
      }
      continue;
    }

    const int32_t first_key = key_indices[0];
    const int64_t first_start = sparse_offsets[first_key];
    const int64_t first_end = sparse_offsets[first_key + 1];
    result.ids.assign(postings + first_start, postings + first_end);
    for (size_t operand = 1;
         operand < key_indices.size() && !result.ids.empty(); ++operand) {
      const int32_t key = key_indices[operand];
      const int32_t dense_row = dense_rows[key];
      if (dense_row >= 0) {
        filter_postings_by_dense_mask<Packed>(
            result.ids,
            dense_masks + static_cast<int64_t>(dense_row) * dense_width);
      } else {
        const int64_t start = sparse_offsets[key];
        const int64_t end = sparse_offsets[key + 1];
        result.ids = intersect_sorted_postings(result.ids, postings + start,
                                               end - start);
      }
    }
  }
}
