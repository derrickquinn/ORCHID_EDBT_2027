#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <faiss/MetricType.h>

namespace orchid::baseline {

namespace py = pybind11;

using FloatMatrix = py::array_t<float, py::array::c_style>;
using ByteMatrix = py::array_t<std::uint8_t, py::array::c_style>;
using IntVector = py::array_t<std::int32_t, py::array::c_style>;

enum class Metric {
  L2,
  INNER_PRODUCT,
};

inline faiss::MetricType to_faiss_metric(Metric metric) {
  switch (metric) {
  case Metric::L2:
    return faiss::METRIC_L2;
  case Metric::INNER_PRODUCT:
    return faiss::METRIC_INNER_PRODUCT;
  }
  throw std::invalid_argument("unknown metric");
}

inline void require_positive(const char *name, py::ssize_t value) {
  if (value <= 0) {
    throw std::invalid_argument(std::string(name) + " must be positive");
  }
}

inline void validate_vectors(const FloatMatrix &vectors) {
  if (vectors.ndim() != 2) {
    throw std::invalid_argument(
        "vectors must be a 2-D C-contiguous float32 array");
  }
  require_positive("number of vectors", vectors.shape(0));
  require_positive("vector dimension", vectors.shape(1));
}

inline void validate_queries(const FloatMatrix &queries,
                             py::ssize_t dimension) {
  if (queries.ndim() != 2) {
    throw std::invalid_argument(
        "queries must be a 2-D C-contiguous float32 array");
  }
  if (queries.shape(1) != dimension) {
    throw std::invalid_argument("query dimension does not match the index");
  }
}

inline void validate_masks(const ByteMatrix &masks, py::ssize_t query_count,
                           py::ssize_t vector_count) {
  if (masks.ndim() != 2) {
    throw std::invalid_argument("masks must be a 2-D C-contiguous uint8 array");
  }
  if (masks.shape(0) != query_count || masks.shape(1) != vector_count) {
    throw std::invalid_argument(
        "masks must have shape (number of queries, index.ntotal)");
  }
}

inline int checked_int(const char *name, py::ssize_t value) {
  require_positive(name, value);
  if (value > std::numeric_limits<int>::max()) {
    throw std::overflow_error(std::string(name) + " does not fit in an int");
  }
  return static_cast<int>(value);
}

inline void bind_common(py::module_ &module) {
  py::enum_<Metric>(module, "Metric", py::module_local())
      .value("L2", Metric::L2)
      .value("INNER_PRODUCT", Metric::INNER_PRODUCT)
      .export_values();
  module.attr("baseline_revision") = ORCHID_BASELINE_REVISION;
}

} // namespace orchid::baseline
