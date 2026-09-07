#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <cuvs/core/bitset.hpp>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/common.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <rmm/device_uvector.hpp>

namespace py = pybind11;
namespace cagra = cuvs::neighbors::cagra;

namespace {

constexpr const char *kArtifactRepository =
    "https://github.com/nicolelii/CAGRA_SIGMOD_Artifacts";

enum class Metric { L2, INNER_PRODUCT };

void cuda_check(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

cuvs::distance::DistanceType distance_type(Metric metric) {
  switch (metric) {
  case Metric::L2:
    return cuvs::distance::DistanceType::L2Expanded;
  case Metric::INNER_PRODUCT:
    return cuvs::distance::DistanceType::InnerProduct;
  }
  throw std::runtime_error("unsupported CAGRA metric");
}

class Index {
public:
  using NativeIndex = cagra::index<float, uint32_t>;
  using FloatArray = py::array_t<float, py::array::c_style>;
  using MaskArray = py::array_t<uint8_t, py::array::c_style>;

  static std::unique_ptr<Index> build(FloatArray vectors, size_t graph_degree,
                                      size_t intermediate_graph_degree,
                                      int device, Metric metric) {
    validate_vectors(vectors);
    if (graph_degree == 0 || intermediate_graph_degree == 0) {
      throw std::invalid_argument("CAGRA graph degrees must be positive");
    }
    if (intermediate_graph_degree < graph_degree) {
      throw std::invalid_argument(
          "intermediate_graph_degree must be at least graph_degree");
    }

    auto result = create(device, metric);
    cagra::index_params params;
    params.metric = distance_type(metric);
    params.graph_degree = graph_degree;
    params.intermediate_graph_degree = intermediate_graph_degree;

    const auto rows = static_cast<int64_t>(vectors.shape(0));
    const auto columns = static_cast<int64_t>(vectors.shape(1));
    auto host_vectors =
        raft::make_host_matrix_view<const float, int64_t, raft::row_major>(
            vectors.data(), rows, columns);
    {
      py::gil_scoped_release release;
      result->select_device();
      result->index_ = std::make_unique<NativeIndex>(
          cagra::build(result->resources_, params, host_vectors));
      result->synchronize();
    }
    return result;
  }

  static std::unique_ptr<Index> load(const std::string &path, int device) {
    auto result = create(device, Metric::L2);
    {
      py::gil_scoped_release release;
      result->select_device();
      cagra::deserialize(result->resources_, path, result->index_.get());
      result->synchronize();
    }
    if (result->index_->size() == 0 || result->index_->dim() == 0) {
      throw std::runtime_error("deserialized CAGRA index is empty");
    }
    return result;
  }

  void save(const std::string &path) {
    py::gil_scoped_release release;
    select_device();
    cagra::serialize(resources_, path, *index_, true);
    synchronize();
  }

  py::tuple search(FloatArray queries, MaskArray masks, size_t k,
                   size_t itopk_size) {
    validate_queries(queries);
    const auto query_count = static_cast<int64_t>(queries.shape(0));
    const auto expected_mask_bytes =
        (static_cast<size_t>(index_->size()) + 7) / 8;
    if (masks.ndim() != 2 || masks.shape(0) != queries.shape(0) ||
        static_cast<size_t>(masks.shape(1)) != expected_mask_bytes) {
      throw std::invalid_argument(
          "masks must be packed uint8[num_queries, ceil(ntotal / 8)]");
    }
    if (k == 0 || itopk_size < k) {
      throw std::invalid_argument(
          "itopk_size must be at least k, and k positive");
    }

    py::array_t<float> distances({query_count, static_cast<int64_t>(k)});
    py::array_t<uint32_t> neighbors({query_count, static_cast<int64_t>(k)});

    {
      py::gil_scoped_release release;
      select_device();
      const auto stream = raft::resource::get_cuda_stream(resources_);
      const auto dimension = static_cast<int64_t>(index_->dim());
      const auto words_per_mask =
          (static_cast<int64_t>(index_->size()) + 31) / 32;
      const auto device_mask_bytes =
          static_cast<size_t>(words_per_mask) * sizeof(uint32_t);

      ensure_buffer(device_queries_,
                    static_cast<size_t>(query_count * dimension), stream);
      ensure_buffer(device_masks_,
                    static_cast<size_t>(query_count * words_per_mask), stream);
      ensure_buffer(device_neighbors_, static_cast<size_t>(query_count) * k,
                    stream);
      ensure_buffer(device_distances_, static_cast<size_t>(query_count) * k,
                    stream);

      cuda_check(cudaMemcpyAsync(device_queries_->data(), queries.data(),
                                 static_cast<size_t>(query_count * dimension) *
                                     sizeof(float),
                                 cudaMemcpyHostToDevice, stream.value()),
                 "copying CAGRA queries to the GPU");
      cuda_check(
          cudaMemsetAsync(device_masks_->data(), 0,
                          static_cast<size_t>(query_count) * device_mask_bytes,
                          stream.value()),
          "clearing CAGRA device masks");
      cuda_check(cudaMemcpy2DAsync(device_masks_->data(), device_mask_bytes,
                                   masks.data(), expected_mask_bytes,
                                   expected_mask_bytes,
                                   static_cast<size_t>(query_count),
                                   cudaMemcpyHostToDevice, stream.value()),
                 "copying CAGRA masks to the GPU");

      cagra::search_params params;
      params.itopk_size = itopk_size;
      params.algo = cagra::search_algo::MULTI_CTA;
      // This matches the collaborator artifact: the bitmap remains active,
      // while CAGRA does not inflate iTopK based on the filtering rate.
      params.filtering_rate = 0.0f;

      // The artifact calls CAGRA once per query because bitset_filter carries
      // one bitmap. Predicate evaluation remains batched by the outer harness.
      for (int64_t query = 0; query < query_count; ++query) {
        auto query_view = raft::make_device_matrix_view<const float, int64_t,
                                                        raft::row_major>(
            device_queries_->data() + query * dimension, 1, dimension);
        auto neighbor_view =
            raft::make_device_matrix_view<uint32_t, int64_t, raft::row_major>(
                device_neighbors_->data() + query * static_cast<int64_t>(k), 1,
                k);
        auto distance_view =
            raft::make_device_matrix_view<float, int64_t, raft::row_major>(
                device_distances_->data() + query * static_cast<int64_t>(k), 1,
                k);
        auto *mask_words = device_masks_->data() +
                           query * static_cast<int64_t>(words_per_mask);
        cuvs::core::bitset_view<uint32_t, int64_t> bitset(
            mask_words, static_cast<int64_t>(index_->size()));
        cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t> filter(
            bitset);
        cagra::search(resources_, params, *index_, query_view, neighbor_view,
                      distance_view, filter);
        // Preserve the artifact's batch-size-one execution semantics.
        synchronize();
      }

      cuda_check(cudaMemcpyAsync(
                     neighbors.mutable_data(), device_neighbors_->data(),
                     static_cast<size_t>(query_count) * k * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost, stream.value()),
                 "copying CAGRA neighbors to the host");
      cuda_check(
          cudaMemcpyAsync(distances.mutable_data(), device_distances_->data(),
                          static_cast<size_t>(query_count) * k * sizeof(float),
                          cudaMemcpyDeviceToHost, stream.value()),
          "copying CAGRA distances to the host");
      synchronize();
    }
    return py::make_tuple(std::move(distances), std::move(neighbors));
  }

  size_t size() const { return static_cast<size_t>(index_->size()); }
  size_t dimension() const { return static_cast<size_t>(index_->dim()); }

private:
  template <typename T>
  static void ensure_buffer(std::unique_ptr<rmm::device_uvector<T>> &buffer,
                            size_t required, rmm::cuda_stream_view stream) {
    if (!buffer || buffer->size() < required) {
      buffer = std::make_unique<rmm::device_uvector<T>>(required, stream);
    }
  }

  Index(int device, Metric metric)
      : device_(device), resources_(),
        index_(
            std::make_unique<NativeIndex>(resources_, distance_type(metric))) {}

  static std::unique_ptr<Index> create(int device, Metric metric) {
    if (device < 0) {
      throw std::invalid_argument("CUDA device must be non-negative");
    }
    cuda_check(cudaSetDevice(device), "selecting the CAGRA CUDA device");
    return std::unique_ptr<Index>(new Index(device, metric));
  }

  static void validate_vectors(const FloatArray &vectors) {
    if (vectors.ndim() != 2 || vectors.shape(0) <= 0 || vectors.shape(1) <= 0) {
      throw std::invalid_argument(
          "vectors must be a nonempty C-contiguous float32 matrix");
    }
    if (static_cast<uint64_t>(vectors.shape(0)) >
        static_cast<uint64_t>(UINT32_MAX)) {
      throw std::invalid_argument("CAGRA supports at most UINT32_MAX vectors");
    }
  }

  void validate_queries(const FloatArray &queries) const {
    if (queries.ndim() != 2 || queries.shape(0) <= 0 ||
        static_cast<size_t>(queries.shape(1)) != dimension()) {
      throw std::invalid_argument(
          "queries must be a compatible C-contiguous float32 matrix");
    }
  }

  void select_device() const {
    cuda_check(cudaSetDevice(device_), "selecting the CAGRA CUDA device");
  }

  void synchronize() const {
    const auto stream = raft::resource::get_cuda_stream(resources_);
    cuda_check(cudaStreamSynchronize(stream.value()),
               "synchronizing the CAGRA CUDA stream");
  }

  int device_;
  raft::resources resources_;
  std::unique_ptr<NativeIndex> index_;
  std::unique_ptr<rmm::device_uvector<float>> device_queries_;
  std::unique_ptr<rmm::device_uvector<uint32_t>> device_masks_;
  std::unique_ptr<rmm::device_uvector<uint32_t>> device_neighbors_;
  std::unique_ptr<rmm::device_uvector<float>> device_distances_;
};

} // namespace

PYBIND11_MODULE(_cagra_native, module) {
  module.doc() =
      "Native cuVS filtered-CAGRA adapter derived from the SIGMOD artifact";
  module.attr("artifact_repository") = kArtifactRepository;
  module.attr("artifact_revision") = ORCHID_CAGRA_ARTIFACT_REVISION;

  py::enum_<Metric>(module, "Metric")
      .value("L2", Metric::L2)
      .value("INNER_PRODUCT", Metric::INNER_PRODUCT);

  py::class_<Index>(module, "Index")
      .def_static("build", &Index::build, py::arg("vectors"),
                  py::arg("graph_degree") = 32,
                  py::arg("intermediate_graph_degree") = 64,
                  py::arg("device") = 0, py::arg("metric") = Metric::L2)
      .def_static("load", &Index::load, py::arg("path"), py::arg("device") = 0)
      .def("save", &Index::save, py::arg("path"))
      .def("search", &Index::search, py::arg("queries"), py::arg("masks"),
           py::arg("k"), py::arg("ef_search"))
      .def_property_readonly("ntotal", &Index::size)
      .def_property_readonly("d", &Index::dimension);
}
