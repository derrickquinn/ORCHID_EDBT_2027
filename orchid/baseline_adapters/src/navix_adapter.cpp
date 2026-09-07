#include "adapter_common.h"

#include <memory>
#include <stdexcept>
#include <string>

#include <faiss/IndexHNSW.h>
#include <faiss/impl/HNSW.h>
#include <faiss/index_io.h>

namespace orchid::baseline {

class NavixIndex {
public:
  static std::unique_ptr<NavixIndex> build(const FloatMatrix &vectors, int m,
                                           int ef_construction, Metric metric) {
    validate_vectors(vectors);
    require_positive("m", m);
    require_positive("ef_construction", ef_construction);
    const int dimension = checked_int("vector dimension", vectors.shape(1));

    auto result = std::unique_ptr<NavixIndex>(new NavixIndex());
    {
      py::gil_scoped_release release;
      auto index = std::make_unique<faiss::IndexHNSWFlat>(
          dimension, m, to_faiss_metric(metric));
      index->hnsw.efConstruction = ef_construction;
      index->add(vectors.shape(0), vectors.data());
      result->index_ = std::move(index);
    }
    return result;
  }

  static std::unique_ptr<NavixIndex> load(const std::string &path) {
    std::unique_ptr<faiss::Index> loaded;
    {
      py::gil_scoped_release release;
      loaded.reset(faiss::read_index(path.c_str()));
    }
    auto *typed = dynamic_cast<faiss::IndexHNSW *>(loaded.get());
    if (typed == nullptr) {
      throw std::invalid_argument("file does not contain a NaviX HNSW index");
    }
    loaded.release();
    return std::unique_ptr<NavixIndex>(
        new NavixIndex(std::unique_ptr<faiss::IndexHNSW>(typed)));
  }

  void save(const std::string &path) const {
    py::gil_scoped_release release;
    faiss::write_index(index_.get(), path.c_str());
  }

  py::tuple search(const FloatMatrix &queries, const ByteMatrix &masks, int k,
                   int ef_search) const {
    validate_queries(queries, index_->d);
    validate_masks(masks, queries.shape(0), index_->ntotal);
    require_positive("k", k);
    require_positive("ef_search", ef_search);

    py::array_t<float> distances(
        {queries.shape(0), static_cast<py::ssize_t>(k)});
    py::array_t<faiss::idx_t> labels(
        {queries.shape(0), static_cast<py::ssize_t>(k)});
    faiss::SearchParametersHNSW parameters;
    parameters.efSearch = ef_search;
    // The pinned fork's navix_search computes efSearch from parameters, but
    // navix_hybrid_search reads the index field directly. Keep the adapter's
    // public parameter effective without carrying a source patch to the fork.
    index_->hnsw.efSearch = ef_search;
    {
      py::gil_scoped_release release;
      index_->navix_search(queries.shape(0), queries.data(), k,
                           distances.mutable_data(), labels.mutable_data(),
                           reinterpret_cast<const char *>(masks.data()),
                           &parameters);
    }
    return py::make_tuple(std::move(distances), std::move(labels));
  }

  faiss::idx_t ntotal() const { return index_->ntotal; }

  int dimension() const { return index_->d; }

private:
  NavixIndex() = default;

  explicit NavixIndex(std::unique_ptr<faiss::IndexHNSW> index)
      : index_(std::move(index)) {}

  std::unique_ptr<faiss::IndexHNSW> index_;
};

} // namespace orchid::baseline

PYBIND11_MODULE(navix_adapter, module) {
  using namespace orchid::baseline;
  module.doc() = "Zero-copy byte-mask adapter for the pinned NaviX baseline";
  bind_common(module);
  py::class_<NavixIndex>(module, "Index", py::module_local())
      .def_static("build", &NavixIndex::build, py::arg("vectors").noconvert(),
                  py::arg("m") = 32, py::arg("ef_construction") = 40,
                  py::arg("metric") = Metric::L2)
      .def_static("load", &NavixIndex::load, py::arg("path"))
      .def("save", &NavixIndex::save, py::arg("path"))
      .def("search", &NavixIndex::search, py::arg("queries").noconvert(),
           py::arg("masks").noconvert(), py::arg("k"), py::arg("ef_search"))
      .def_property_readonly("ntotal", &NavixIndex::ntotal)
      .def_property_readonly("d", &NavixIndex::dimension);
}
