#include "adapter_common.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <faiss/IndexACORN.h>
#include <faiss/impl/ACORN.h>
#include <faiss/index_io.h>

namespace orchid::baseline {

class AcornIndex {
public:
  static std::unique_ptr<AcornIndex> build(const FloatMatrix &vectors,
                                           const IntVector &metadata, int m,
                                           int gamma, int m_beta,
                                           int ef_construction, Metric metric) {
    static_assert(sizeof(int) == sizeof(std::int32_t));
    validate_vectors(vectors);
    if (metadata.ndim() != 1 || metadata.shape(0) != vectors.shape(0)) {
      throw std::invalid_argument("metadata must be a 1-D C-contiguous int32 "
                                  "array with one value per vector");
    }
    require_positive("m", m);
    require_positive("gamma", gamma);
    require_positive("m_beta", m_beta);
    require_positive("ef_construction", ef_construction);
    const int dimension = checked_int("vector dimension", vectors.shape(1));

    auto result = std::unique_ptr<AcornIndex>(new AcornIndex());
    result->metadata_.assign(metadata.data(),
                             metadata.data() + metadata.size());
    {
      py::gil_scoped_release release;
      auto index = std::make_unique<faiss::IndexACORNFlat>(
          dimension, m, gamma, result->metadata_, m_beta,
          to_faiss_metric(metric));
      index->acorn.efConstruction = ef_construction;
      index->add(vectors.shape(0), vectors.data());
      result->index_ = std::move(index);
    }
    return result;
  }

  static std::unique_ptr<AcornIndex> load(const std::string &path) {
    std::unique_ptr<faiss::Index> loaded;
    {
      py::gil_scoped_release release;
      loaded.reset(faiss::read_index(path.c_str()));
    }
    auto *typed = dynamic_cast<faiss::IndexACORN *>(loaded.get());
    if (typed == nullptr) {
      throw std::invalid_argument("file does not contain an ACORN index");
    }
    loaded.release();
    return std::unique_ptr<AcornIndex>(
        new AcornIndex(std::unique_ptr<faiss::IndexACORN>(typed)));
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
    faiss::SearchParametersACORN parameters;
    parameters.efSearch = ef_search;
    // The pinned fork passes SearchParametersACORN to its expansion loop, but
    // hybrid_search sizes the candidate heap from the index field directly.
    index_->acorn.efSearch = ef_search;
    {
      py::gil_scoped_release release;
      index_->search(
          queries.shape(0), queries.data(), k, distances.mutable_data(),
          labels.mutable_data(),
          reinterpret_cast<char *>(const_cast<std::uint8_t *>(masks.data())),
          &parameters);
    }
    return py::make_tuple(std::move(distances), std::move(labels));
  }

  faiss::idx_t ntotal() const { return index_->ntotal; }

  int dimension() const { return index_->d; }

private:
  AcornIndex() = default;

  explicit AcornIndex(std::unique_ptr<faiss::IndexACORN> index)
      : metadata_(static_cast<std::size_t>(index->ntotal), 0),
        index_(std::move(index)) {
    // The pinned fork does not serialize metadata and read_index leaves a
    // dangling pointer. Search is mask-driven, but keep the pointer valid
    // for every code path (including its debug/statistics paths).
    index_->acorn.metadata = metadata_.data();
  }

  // Keep this before index_: reverse destruction order destroys the index
  // before releasing the storage backing its raw metadata pointer.
  std::vector<int> metadata_;
  std::unique_ptr<faiss::IndexACORN> index_;
};

} // namespace orchid::baseline

PYBIND11_MODULE(acorn_adapter, module) {
  using namespace orchid::baseline;
  module.doc() = "Zero-copy byte-mask adapter for the pinned ACORN baseline";
  bind_common(module);
  py::class_<AcornIndex>(module, "Index", py::module_local())
      .def_static("build", &AcornIndex::build, py::arg("vectors").noconvert(),
                  py::arg("metadata").noconvert(), py::arg("m") = 32,
                  py::arg("gamma") = 12, py::arg("m_beta") = 64,
                  py::arg("ef_construction") = 40,
                  py::arg("metric") = Metric::L2)
      .def_static("load", &AcornIndex::load, py::arg("path"))
      .def("save", &AcornIndex::save, py::arg("path"))
      .def("search", &AcornIndex::search, py::arg("queries").noconvert(),
           py::arg("masks").noconvert(), py::arg("k"), py::arg("ef_search"))
      .def_property_readonly("ntotal", &AcornIndex::ntotal)
      .def_property_readonly("d", &AcornIndex::dimension);
}
