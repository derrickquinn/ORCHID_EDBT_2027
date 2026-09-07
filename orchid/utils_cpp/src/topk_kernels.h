#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

template <int K>
static inline void topk_row_fixed(const float* row, int32_t width,
                                  int64_t* out_ids) {
  float values[K];
  int64_t ids[K];
  for (int32_t rank = 0; rank < K; ++rank) {
    values[rank] = -std::numeric_limits<float>::infinity();
    ids[rank] = std::numeric_limits<int64_t>::max();
  }

  for (int32_t column = 0; column < width; ++column) {
    const float value = row[column];
    const int64_t id = column;
    if (value < values[K - 1] ||
        (value == values[K - 1] && id >= ids[K - 1])) {
      continue;
    }

    int32_t position = K - 1;
    while (position > 0 &&
           (value > values[position - 1] ||
            (value == values[position - 1] && id < ids[position - 1]))) {
      values[position] = values[position - 1];
      ids[position] = ids[position - 1];
      --position;
    }
    values[position] = value;
    ids[position] = id;
  }

  for (int32_t rank = 0; rank < K; ++rank) {
    out_ids[rank] = ids[rank];
  }
}

static inline void topk_row_bounded(const float* row, int32_t width,
                                    int32_t k, int64_t* out_ids) {
  constexpr int32_t kMaxTopK = 64;
  float values[kMaxTopK];
  int64_t ids[kMaxTopK];
  for (int32_t rank = 0; rank < k; ++rank) {
    values[rank] = -std::numeric_limits<float>::infinity();
    ids[rank] = std::numeric_limits<int64_t>::max();
  }

  for (int32_t column = 0; column < width; ++column) {
    const float value = row[column];
    const int64_t id = column;
    if (value < values[k - 1] ||
        (value == values[k - 1] && id >= ids[k - 1])) {
      continue;
    }

    int32_t position = k - 1;
    while (position > 0 &&
           (value > values[position - 1] ||
            (value == values[position - 1] && id < ids[position - 1]))) {
      values[position] = values[position - 1];
      ids[position] = ids[position - 1];
      --position;
    }
    values[position] = value;
    ids[position] = id;
  }

  for (int32_t rank = 0; rank < k; ++rank) {
    out_ids[rank] = ids[rank];
  }
}

struct TopKCandidate {
  float value;
  int32_t id;
};

struct TopKCandidateBetter {
  bool operator()(const TopKCandidate& left,
                  const TopKCandidate& right) const {
    return left.value > right.value ||
           (left.value == right.value && left.id < right.id);
  }
};

static inline void topk_row_partition(const float* row, int32_t width,
                                      int32_t k, int64_t* out_ids) {
  std::vector<TopKCandidate> candidates(static_cast<size_t>(width));
  for (int32_t column = 0; column < width; ++column) {
    candidates[column] = {row[column], column};
  }

  const TopKCandidateBetter better;
  auto topk_end = candidates.begin() + k;
  if (k < width) {
    std::nth_element(candidates.begin(), topk_end, candidates.end(), better);
  }
  std::sort(candidates.begin(), topk_end, better);
  for (int32_t rank = 0; rank < k; ++rank) {
    out_ids[rank] = candidates[rank].id;
  }
}

static void topk_small_impl(const float* scores, int32_t rows, int32_t width,
                            int32_t k, int64_t* out_ids, int32_t threads) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
  for (int32_t row_index = 0; row_index < rows; ++row_index) {
    const float* row =
        scores + static_cast<int64_t>(row_index) * width;
    int64_t* row_out = out_ids + static_cast<int64_t>(row_index) * k;
    if (k == 4) {
      topk_row_fixed<4>(row, width, row_out);
    } else if (k <= 64) {
      topk_row_bounded(row, width, k, row_out);
    } else {
      topk_row_partition(row, width, k, row_out);
    }
  }
}
