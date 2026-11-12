#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <future>
#include <type_traits>
#include <algorithm>
#include <stdexcept> 
#include <cmath>
#include <memory>
#include <optional>
#include <variant>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <limits>
#include <utility>

#if defined(ENABLE_GPU)
#include "cluser_legals_gpu_kernel.h"
#endif

// This represents the interface in which cuda interacts with python programs.
// Any actual cuda semantics should stay out of this file.

namespace py = pybind11;
using PyFuture = py::object;

class GpuPromise;
namespace detail { template <typename... T> class SpanTemplateUtil; }

// A non owning buffer of Gpu data to be be completed on the Gpu
class GpuBuffer {
public:
    void* m_data{ nullptr };
    size_t m_size{ 0 };
    std::shared_ptr<GpuPromise> m_promise{ nullptr };

    GpuBuffer() = default;
    GpuBuffer(void* data, size_t size, std::shared_ptr<GpuPromise> promise);
};

// A typed span of Gpu data to be completed on the Gpu
template<typename T>
class GpuSpan : public GpuBuffer {
    static_assert(std::is_trivially_copyable<T>::value, "Template parameter T must be a primitive type.");

    bool m_uploaded{ 0 };
    std::vector<py::array_t<T, py::array::c_style>> m_cpuInformation;

    template <typename...> friend class detail::SpanTemplateUtil;
public:
    GpuSpan() = default;
    GpuSpan(void* data, size_t elementCount, std::shared_ptr<GpuPromise> promise);

    // Pybind API
    PyFuture collectToCpu() const;
    
    // Gets underlying buffer, assuming it has already been uploaded.
    const GpuBuffer& getBuffer() const;
};


namespace detail {
    // Helper function for adding GPU span
    template<typename T>
    void pymoduleAddGpuSpan(py::module_ &m, const std::string& name) {
        py::class_<GpuSpan<T>, GpuBuffer>(m, (name + "_gpu_span").c_str())
            .def("collect_to_cpu", &GpuSpan<T>::collectToCpu);
    }

    // Structural string type for C++20 non-type template parameters.
    template <std::size_t N>
    struct FixedString {
        char value[N];

        constexpr FixedString(const char (&text)[N]) {
            std::copy_n(text, N, value);
        }
    };

    template <typename T, FixedString Name>
    struct NamedType {
        using type = T;
        static constexpr auto name = Name;
    };

    // Since GpuSpan is a templated type, this util allows
    // a limited amount of templates to be sent to pybind.
    template <typename... Entries>
    class SpanTemplateUtil {    
    public:
        // Gpu Span upload information

        // Returns the byte count needed to upload a single GPU span.
        static size_t getSpanSizesForUpload(py::handle item) {
            bool isSupportedSpan = false;
            size_t accumulatedSize = 0;
            ([&]{
                if (py::isinstance<GpuSpan<typename Entries::type>>(item)) {
                    GpuSpan<typename Entries::type>& span = item.cast<GpuSpan<typename Entries::type>&>();
                    isSupportedSpan = true;

                    if (span.m_uploaded) {
                        throw std::runtime_error("Spans cannot be reuploaded again");
                    }

                    for (const auto& source : span.m_cpuInformation) {
                        const size_t bytes = static_cast<size_t>(source.nbytes());
                        if (bytes > std::numeric_limits<size_t>::max() - accumulatedSize) {
                            throw std::overflow_error("GPU span size overflow");
                        }
                        accumulatedSize += bytes;
                    }
                }
            }(), ...);
            
            if (!isSupportedSpan) {
                throw std::runtime_error("Uploaded type is not supported");
            }
            return accumulatedSize;
        }

        // Attaches GPU storage without changing the span's element count.
        static std::vector<py::array> setSpanBuffer(
            py::handle item, void* data, std::shared_ptr<GpuPromise> promise) {
            bool successfulSet = false;
            std::vector<py::array> sources;

            (..., ([&]{
                if (!successfulSet && py::isinstance<GpuSpan<typename Entries::type>>(item)) {
                    GpuSpan<typename Entries::type>& span = item.cast<GpuSpan<typename Entries::type>&>();
                    if (span.m_uploaded) {
                        throw std::runtime_error("Spans cannot be reuploaded again");
                    }
                    sources.reserve(span.m_cpuInformation.size());
                    for (auto& source : span.m_cpuInformation) {
                        sources.push_back(std::move(source));
                    }
                    span.m_cpuInformation.clear();
                    span.m_data = data;
                    span.m_promise = std::move(promise);
                    span.m_uploaded = true;
                    successfulSet = true;
                }
            }()));

            if (!successfulSet) {
                throw std::runtime_error("Given item is not a supported span");
            }
            return sources;
        }

        // Gpu span initialization logic

        // Formats object to be uploaded to gpu
        static py::object formatGpuUpload(py::object item) {
            py::list items;
            if (py::isinstance<py::list>(item) || py::isinstance<py::tuple>(item)) {
                items = py::list(item);
            } else {
                items.append(item);
            }
            if (items.empty()) {
                throw std::runtime_error("Cannot infer an empty list's dtype; use a typed NumPy array");
            }

            using Array = std::variant<py::array_t<typename Entries::type, py::array::c_style>...>;
            std::vector<Array> arrays;
            arrays.reserve(items.size());
            for (py::handle value : items) {
                py::object item = py::reinterpret_borrow<py::object>(value);
                // Let NumPy infer scalar/vector dtypes before the supported array checks.
                if (!py::isinstance<py::array>(item)) {
                    py::array array = py::array::ensure(item);
                    if (array) item = std::move(array);
                }
                std::optional<Array> retArray;

                // Gets a shallow copy if it already is in c style
                (..., ([&]{
                    if (py::isinstance<py::array_t<typename Entries::type>>(item)) {
                        py::array arr = py::reinterpret_borrow<py::array>(item);
                        if (arr.flags() & py::array::c_style) {
                            retArray = py::reinterpret_borrow<py::array_t<typename Entries::type, py::array::c_style>>(item);
                        }
                    }
                }()));

                // Casts and causes a deep copy if non c style
                if (!retArray.has_value()) {
                    (..., ([&]{
                        if (py::isinstance<py::array_t<typename Entries::type>>(item))
                            retArray = item.cast<py::array_t<typename Entries::type, py::array::c_style>>();
                    }()));
                }
                if (!retArray.has_value()) {
                    throw std::runtime_error("Uploaded type is not supported");
                }
                arrays.push_back(std::move(*retArray));
            }

            // Extract a corresponding span from the cpu data type
            py::object result = py::none();
            bool hasAddedSpan = false;
            (..., ([&]{
                using CpuArray = py::array_t<typename Entries::type, py::array::c_style>;
                if (!hasAddedSpan && std::get_if<CpuArray>(&arrays.front())) {
                    GpuSpan<typename Entries::type> retSpan{ };
                    retSpan.m_uploaded = false;
                    retSpan.m_cpuInformation.reserve(arrays.size());
                    for (auto& array : arrays) {
                        auto* ptr = std::get_if<CpuArray>(&array);
                        if (!ptr) {
                            throw std::runtime_error("All vectors in a GPU span must have the same dtype");
                        }
                        retSpan.m_size += static_cast<size_t>(ptr->size());
                        retSpan.m_cpuInformation.push_back(std::move(*ptr));
                    }
                    result = py::cast(std::move(retSpan));
                    hasAddedSpan = true;
                }
            }()));

            if (result.is_none()) {
                throw std::runtime_error("Unreachable point in formatGpuUpload, should have already uploaded");
            }
            
            return result;
        }

        // Gpu span binding logic 

        static void addAllSupportedGpuSpan(py::module_ &m) {
            ([&]{
                pymoduleAddGpuSpan<typename Entries::type>(m, Entries::name.value);
            }(), ...);
        }
    };

    // Extract a type's written name from the macro's stringified argument list.
    template <FixedString Names, std::size_t Index>
    constexpr auto gpuSpanTypeName() {
        std::size_t begin = 0;
        for (std::size_t i = 0; i < Index; ++i) {
            while (Names.value[begin] != ',' && Names.value[begin] != '\0') ++begin;
            if (Names.value[begin] == ',') ++begin;
        }
        const auto isSpace = [](char c) {
            return c == ' ' || c == '\t' || c == '\n' || c == '\r';
        };
        while (isSpace(Names.value[begin])) ++begin;
        std::size_t end = begin;
        while (Names.value[end] != ',' && Names.value[end] != '\0') ++end;
        while (end > begin && isSpace(Names.value[end - 1])) --end;

        auto name = Names;
        std::fill_n(name.value, sizeof(name.value), '\0');
        std::copy_n(Names.value + begin, end - begin, name.value);
        return name;
    }

    template <FixedString Names, typename... Types>
    struct GpuSpanTypes {
        template <std::size_t... Indices>
        static auto make(std::index_sequence<Indices...>)
            -> SpanTemplateUtil<NamedType<Types, gpuSpanTypeName<Names, Indices>()>...>;

        using type = decltype(make(std::index_sequence_for<Types...>{}));
    };

    // Preserve the spelling of each type, including typedefs, as its Python name.
    // Example: using u8 = uint8_t; ORCHID_GPU_SPAN_UTIL(MySpans, u8, int32_t);
    #define ORCHID_GPU_SPAN_UTIL(Alias, ...) \
        typedef typename ::detail::GpuSpanTypes<#__VA_ARGS__, __VA_ARGS__>::type Alias

    using byte = uint8_t;
    using u32 = uint32_t;
    using i32 = int32_t;
    using i64 = int64_t;
    ORCHID_GPU_SPAN_UTIL(SpanUtil, byte, u32, i32, i64, float);
}


// Hidden implementation of cuda specific behavior
class GpuPromiseData;
class ReadbackPromiseData;
struct StreamCapsule;

// Retains the original arrays; no additional host buffer is allocated.
struct CpuUploadCapsule {
    std::vector<py::array> sources;
};

// A set of Gpu data that may or may not be in transit
class GpuPromise : public std::enable_shared_from_this<GpuPromise> {
public:
    // Python upload owners are retained until queued copies have completed.
    std::vector<py::array> m_uploadSources;

    // >>> C++/Cuda exclusive <<<
    std::shared_ptr<GpuPromiseData> m_data;

    GpuPromise(std::shared_ptr<GpuPromiseData> data);
    ~GpuPromise();
    const GpuPromise& operator=(const GpuPromise& promise) = delete;

    // >>> Pybind <<<
    GpuSpan<uint8_t> collectBufferAsync();
    GpuSpan<uint8_t> collectBufferSync();

    PyFuture getStartTimeStamp();
    PyFuture getEndTimeStamp();
    // Makes an external stream (raw integer handle, e.g. from FAISS) wait on this promise.
    void makeStreamWait(int64_t stream);
};

// Prepares a buffer of numberic values to be uploaded to the gpu
// Doesn't upload the buffer just yet prepares data to be uploaded
py::object gpuPrep_out_py(py::object item);

// Actually collects the buffers to be sent together in a single GPU promise.
std::shared_ptr<GpuPromise> uploadToGpu_out_py(py::args args);

#if defined(ENABLE_GPU)

#include <vector>
namespace detail {
    // Takes a non owning pointer and creates python object
    typedef std::function<py::object(const void*, size_t)> BufferProcessor;
    // Creates python future from in transit cpu data
    inline PyFuture makeReadbackFuture(std::shared_ptr<const ReadbackPromiseData> data, BufferProcessor processor) {
        py::object loop = py::module_::import("asyncio").attr("get_running_loop")();
        PyFuture fut  = loop.attr("create_future")();

        auto* state = new std::pair<py::object, PyFuture>(loop, fut);

        std::thread([data, processor, state] {
            data->sync();
            {
                py::gil_scoped_acquire gil;
                auto& [loop, fut] = *state;
                loop.attr("call_soon_threadsafe")(fut.attr("set_result"), processor(data->getData(), data->getSize()));
                delete state;
            }
        }).detach();

        return fut;
    }
    inline PyFuture makeTimeStampFuture(TimeStampData data) {
        py::object loop = py::module_::import("asyncio").attr("get_running_loop")();
        PyFuture fut  = loop.attr("create_future")();

        auto* state = new std::pair<py::object, PyFuture>(loop, fut);

        std::thread([data, state] {
            data.sync();
            {
                py::gil_scoped_acquire gil;
                auto& [loop, fut] = *state;
                loop.attr("call_soon_threadsafe")(fut.attr("set_result"), data.getTimeStamp());
                delete state;
            }
        }).detach();

        return fut;
    }

    template<typename T>
    py::object convertSpanData(const void* buffer, size_t size) {
        if (size % sizeof(T) != 0) {
            throw std::runtime_error("Data size is not a multiple of the size of the type.");
        }
        return py::array_t<T, py::array::c_style>(
            static_cast<py::ssize_t>(size / sizeof(T)),
            static_cast<const T*>(buffer));  
    }

    template<typename T>
    py::object convertData(const void* buffer, size_t size) {
        if (size != sizeof(T)) {
            throw std::runtime_error("Data is wrong size for type.");
        }
        T value;
        std::memcpy(&value, buffer, sizeof(T));
        return py::cast(value);
    }
    
    inline std::vector<const PromiseData*> gatherPromises(std::vector<const GpuBuffer*> buffers) {
        std::vector<const PromiseData*> ret;
        for (const GpuBuffer* buffer : buffers) {
            if (!buffer->m_promise || !buffer->m_promise->m_data) {
                throw std::runtime_error("Span has not been uploaded to the GPU yet");
            }
            const PromiseData* promise = buffer->m_promise->m_data.get();
            if (std::find(ret.begin(), ret.end(), promise) == ret.end()) {
                ret.push_back(promise);
            }
        }
        return ret;
    }
}
inline GpuBuffer::GpuBuffer(void* data, size_t size, std::shared_ptr<GpuPromise> promise)
    : m_data(data), m_size(size), m_promise(std::move(promise)) {}
template <typename T>
inline GpuSpan<T>::GpuSpan(void* data, size_t size, std::shared_ptr<GpuPromise> promise)
    : GpuBuffer(data, size, std::move(promise)), m_uploaded(true) {}
inline GpuPromise::GpuPromise(std::shared_ptr<GpuPromiseData> data) : m_data(std::move(data)) {
    if (!m_data) {
        throw std::runtime_error("GPU kernel returned no data");
    }
}
inline GpuPromise::~GpuPromise() {
    // Stream-ordered device frees do not wait for copies from Python arrays.
    if (!m_uploadSources.empty() && m_data) {
        m_data->sync();
    }
}

template <typename T>
PyFuture GpuSpan<T>::collectToCpu() const {
    if (!m_uploaded) {
        throw std::runtime_error("Span has not been uploaded to the GPU yet");
    }
    if (m_data == nullptr && m_size != 0) {
        throw std::runtime_error("Span not given valid data");
    }

    if (!m_promise) {
        throw std::runtime_error("Span has no GPU promise");
    }
    auto readback = m_promise->m_data->copyToCpu(m_data, m_size * sizeof(T));
    return detail::makeReadbackFuture(readback, detail::convertSpanData<T>);
}

template <typename T>
inline const GpuBuffer& GpuSpan<T>::getBuffer() const {
    if (!m_uploaded) {
        throw std::runtime_error("Span has not been uploaded to the GPU yet");
    }
    return *this;
}
inline GpuSpan<uint8_t> GpuPromise::collectBufferAsync() {
    return GpuSpan<uint8_t>(m_data->getData(), m_data->getSize(), shared_from_this());
}
inline GpuSpan<uint8_t> GpuPromise::collectBufferSync() {
    m_data->sync();
    return collectBufferAsync();
}
inline PyFuture GpuPromise::getStartTimeStamp() {
    return detail::makeTimeStampFuture(m_data->getStartTimeStamp());
}
inline PyFuture GpuPromise::getEndTimeStamp() {
    return detail::makeTimeStampFuture(m_data->getEndTimeStamp());
}
inline void GpuPromise::makeStreamWait(int64_t stream) {
    m_data->makeStreamWait(stream);
}
inline PyFuture gpuPrep_out_py(py::object item) {
    return detail::SpanUtil::formatGpuUpload(std::move(item));
}

inline std::shared_ptr<GpuPromise> uploadToGpu_out_py(py::args args) {
    size_t argLength = args.size();
    std::vector<py::handle> dataHandles;
    dataHandles.reserve(argLength);
    std::vector<size_t> dataSizes;
    dataSizes.reserve(argLength);
    size_t cumulative_size = 0;

    // Ensures item sizes are properly accounted for
    for (py::handle item : args) {
        dataHandles.push_back(item);
        size_t itemSize = detail::SpanUtil::getSpanSizesForUpload(item);
        dataSizes.push_back(itemSize);
        const size_t maxSize = std::numeric_limits<size_t>::max();
        if (itemSize > maxSize - (GpuAlignment - 1)) {
            throw std::overflow_error("GPU span size overflow");
        }
        size_t alignedSize = roundToAlignment(itemSize);
        if (alignedSize > maxSize - cumulative_size) {
            throw std::overflow_error("GPU allocation size overflow");
        }
        cumulative_size += alignedSize;
    }

    if (cumulative_size > std::numeric_limits<size_t>::max() - (GpuAlignment - 1)) {
        throw std::overflow_error("GPU allocation size overflow");
    }

    // Creates a block of data with needed size
    auto gpuData = std::make_shared<GpuPromiseData>(cumulative_size);
    void* rawDataIter = gpuData->getData();
    std::shared_ptr<GpuPromise> promise = std::make_shared<GpuPromise>(std::move(gpuData));
    promise->m_uploadSources.reserve(argLength);
    size_t uploadOffset = 0;

    try {
        // Begins partitioning data along block of data
        for (size_t i = 0; i < argLength; ++i) {
            GpuPromiseData& data = *promise->m_data;
            size_t alignedSize = roundToAlignment(dataSizes[i]);
            if (uploadOffset > data.getSize() || alignedSize > data.getSize() - uploadOffset) {
                throw std::runtime_error("GPU data arena is full");
            }
            auto sources = detail::SpanUtil::setSpanBuffer(dataHandles[i], rawDataIter, promise);
            size_t offset = uploadOffset;
            for (auto& source : sources) {
                // Retain each owner and upload directly, without CPU concatenation.
                // Source arrays must remain unchanged until upload completion.
                const size_t bytes = static_cast<size_t>(source.nbytes());
                promise->m_uploadSources.push_back(std::move(source));
                data.uploadCpuDataAsync(promise->m_uploadSources.back().data(), bytes, offset);
                offset += bytes;
            }
            uploadOffset += alignedSize;

            if (rawDataIter != nullptr) {
                rawDataIter = static_cast<uint8_t*>(rawDataIter) + roundToAlignment(dataSizes[i]);
            }
        }
    } catch (...) {
        // Spans may retain the promise after a failure; finish pending uploads.
        promise->m_data->sync();
        throw;
    }
    promise->m_data->finalizePromise();
    return promise;
}

inline GpuSpan<uint8_t> predicateResultsToBitmapGpu_out_py (
    const GpuSpan<uint32_t> posting_lists,
    const GpuSpan<int32_t> posting_sizes,
    const GpuSpan<int32_t> posting_queries,
    const GpuSpan<uint8_t> dense_masks,
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast> dense_mask_queries,
    int32_t n_queries,
    int32_t n_docs) {
    if (n_queries < 0) throw py::value_error("n_queries must be nonnegative");
    if (n_docs <= 0) throw py::value_error("n_docs must be positive");

    // Span sizes count elements. Check metadata without downloading GPU contents.
    const auto elementCount = [](const auto& span, const char* name) {
        const auto& buffer = span.getBuffer();
        if (!buffer.m_promise || (buffer.m_size != 0 && buffer.m_data == nullptr)) {
            throw py::value_error(std::string(name) + " must have a valid uploaded buffer");
        }
        return buffer.m_size;
    };
    const size_t posting_count = elementCount(posting_lists, "posting_lists");
    const size_t size_count = elementCount(posting_sizes, "posting_sizes");
    const size_t posting_query_count = elementCount(posting_queries, "posting_queries");
    const size_t mask_bytes = elementCount(dense_masks, "dense_masks");
    if (dense_mask_queries.ndim() != 1) {
        throw py::value_error("dense_mask_queries must be one-dimensional");
    }
    const size_t mask_query_count = static_cast<size_t>(dense_mask_queries.size());
    if (size_count != posting_query_count) {
        throw py::value_error("posting_sizes and posting_queries must have the same number of elements");
    }
    const size_t query_count = static_cast<size_t>(n_queries);
    if (posting_query_count > query_count || mask_query_count != query_count - posting_query_count) {
        throw py::value_error("posting_queries and dense_mask_queries must total n_queries elements");
    }
    if (posting_query_count == 0 && posting_count != 0) {
        throw py::value_error("posting_lists must be empty when there are no posting queries");
    }
    const size_t byte_width = n_docs / 8 + (n_docs % 8 != 0);
    if (mask_query_count > std::numeric_limits<size_t>::max() / byte_width ||
        mask_bytes != mask_query_count * byte_width) {
        throw py::value_error("dense_masks must contain ceil(n_docs / 8) bytes per dense query");
    }
    for (size_t i = 0; i < mask_query_count; ++i) {
        if (dense_mask_queries.data()[i] < 0 || dense_mask_queries.data()[i] >= n_queries) {
            throw py::value_error("dense_mask_queries contains a query outside [0, n_queries)");
        }
    }
    auto result = std::make_shared<GpuPromise>(predicate_results_to_bitmap_gpu_impl(
        detail::gatherPromises({&posting_lists, &posting_sizes, &posting_queries, &dense_masks}),
        static_cast<const uint32_t*>(posting_lists.getBuffer().m_data), 
        static_cast<const int32_t*>(posting_sizes.getBuffer().m_data), 
        static_cast<const int32_t*>(posting_queries.getBuffer().m_data),
        static_cast<const uint8_t*>(dense_masks.getBuffer().m_data), 
        dense_mask_queries.data(),
        n_queries, n_docs,
        static_cast<int32_t>(posting_query_count), static_cast<int32_t>(mask_query_count),
        static_cast<int64_t>(posting_count)));
    return result->collectBufferAsync();
}

inline GpuSpan<float> cluster_legals_preordered_packed_gpu_out_py (
    const GpuSpan<uint8_t> mask_packed_buffer,
    int32_t n_queries,
    int32_t n_docs,
    const GpuSpan<int32_t> cluster_offsets,
    const GpuSpan<int32_t> cluster_counts,
    int32_t n_list) {
    

    auto result = std::make_shared<GpuPromise>(cluster_legals_preordered_packed_gpu_impl(
        detail::gatherPromises({&mask_packed_buffer, &cluster_offsets, &cluster_counts}),
        static_cast<const uint8_t*>(mask_packed_buffer.getBuffer().m_data), n_queries, n_docs,
        static_cast<const int32_t*>(cluster_offsets.getBuffer().m_data), 
        static_cast<const int32_t*>(cluster_counts.getBuffer().m_data), 
        n_list));
    return GpuSpan<float>(result->m_data->getData(), result->m_data->getSize() / sizeof(float), result);
}

inline py::tuple get_top_rankings_out_py (
    const GpuSpan<float> clustered_legals,
    const GpuSpan<float> cluster_dist,
    int32_t n_queries,
    int32_t n_list,
    int32_t n_probe,
    float beta,
    float eps) {
    if (n_queries < 0) throw py::value_error("n_queries must be nonnegative");
    if (n_list < 0) throw py::value_error("n_list must be nonnegative");
    if (n_probe <= 0 || n_probe > n_list) {
        throw py::value_error("n_probe must be in [1, n_list]");
    }
    const auto& buffer = clustered_legals.getBuffer();
    const auto& distances = cluster_dist.getBuffer();
    if (buffer.m_size != static_cast<size_t>(n_queries) * static_cast<size_t>(n_list) ||
        distances.m_size != buffer.m_size) {
        throw py::value_error("clustered_legals and cluster_dist must contain n_queries * n_list elements");
    }

    auto result = get_top_rankings_gpu_impl(
        detail::gatherPromises({&clustered_legals, &cluster_dist}),
        static_cast<const float*>(buffer.m_data),
        static_cast<const float*>(distances.m_data),
        n_queries, n_list, n_probe, beta, eps);
    return py::make_tuple(
        detail::makeReadbackFuture(result, detail::convertSpanData<int64_t>),
        detail::makeTimeStampFuture(result->getEndTimeStamp()));
}

#else

// If Gpu logic is not support it should throw on any GPU specific logic 
// that attempts to be called. 
class GpuPromiseData { };

inline GpuBuffer::GpuBuffer(void* data, size_t size, std::shared_ptr<GpuPromise> promise) {
    throw std::runtime_error("GPU backend not compiled");
}
inline GpuPromise::GpuPromise(std::shared_ptr<GpuPromiseData> data) {
    throw std::runtime_error("GPU backend not compiled");
}
inline GpuPromise::~GpuPromise() = default;
template <typename T>
inline PyFuture GpuSpan<T>::collectToCpu() const {
    throw std::runtime_error("GPU backend not compiled");
}
template <typename T>
inline const GpuBuffer& GpuSpan<T>::getBuffer() const {
    throw std::runtime_error("GPU backend not compiled");
}
inline GpuSpan<uint8_t> GpuPromise::collectBufferAsync() {
    throw std::runtime_error("GPU backend not compiled");
}
inline GpuSpan<uint8_t> GpuPromise::collectBufferSync() {
    throw std::runtime_error("GPU backend not compiled");
}
inline PyFuture GpuPromise::getStartTimeStamp() {
    throw std::runtime_error("GPU backend not compiled");
}
inline PyFuture GpuPromise::getEndTimeStamp() {
    throw std::runtime_error("GPU backend not compiled");
}
inline void GpuPromise::makeStreamWait(int64_t stream) {
    throw std::runtime_error("GPU backend not compiled");
}
inline py::object gpuPrep_out_py(py::object item) {
    throw std::runtime_error("GPU backend not compiled");
}
inline std::shared_ptr<GpuPromise> uploadToGpu_out_py(py::args args) {
    throw std::runtime_error("GPU backend not compiled");
}

inline GpuSpan<uint8_t> predicateResultsToBitmapGpu_out_py (
    const GpuSpan<uint32_t> posting_lists,
    const GpuSpan<int32_t> posting_sizes,
    const GpuSpan<int32_t> posting_queries,
    const GpuSpan<uint8_t> dense_masks,
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast> mask_queries,
    int32_t n_queries,
    int32_t n_docs) {
    throw std::runtime_error("GPU backend not compiled");
}

inline GpuSpan<float> cluster_legals_preordered_packed_gpu_out_py (
    const GpuSpan<uint8_t> mask_packed_buffer,
    int32_t n_queries,
    int32_t n_docs,
    const GpuSpan<int32_t> cluster_offsets,
    const GpuSpan<int32_t> cluster_counts,
    int32_t n_list) {
    throw std::runtime_error("GPU backend not compiled");
}

inline py::tuple get_top_rankings_out_py (
    const GpuSpan<float> clustered_legals,
    const GpuSpan<float> cluster_dist,
    int32_t n_queries,
    int32_t n_list,
    int32_t n_probe,
    float beta,
    float eps) {
    throw std::runtime_error("GPU backend not compiled");
}

#endif

inline void addGpuModule(py::module_ &m) {
    py::class_<GpuBuffer>(m, "GpuBuffer")
        .def_readonly("size", &GpuBuffer::m_size)
        .def_readonly("promise", &GpuBuffer::m_promise)
        .def_property_readonly("data_ptr", [](const GpuBuffer& b) {
            return static_cast<int64_t>(reinterpret_cast<uintptr_t>(b.m_data));
        });

    py::class_<GpuPromise, std::shared_ptr<GpuPromise>>(m, "GpuPromise")
        .def("collect_buffer_async", &GpuPromise::collectBufferAsync)
        .def("collect_buffer_sync", &GpuPromise::collectBufferSync)
        .def("get_start_time_stamp", &GpuPromise::getStartTimeStamp)
        .def("get_end_time_stamp", &GpuPromise::getEndTimeStamp)
        .def("make_stream_wait", &GpuPromise::makeStreamWait, py::arg("stream"));
    detail::SpanUtil::addAllSupportedGpuSpan(m);
    m.def("prep", &gpuPrep_out_py, py::arg("item"));
    m.def("upload", &uploadToGpu_out_py);
    m.def("predicate_results_to_bitmap", &predicateResultsToBitmapGpu_out_py,
          py::arg("posting_lists"), py::arg("posting_sizes"), py::arg("posting_queries"),
          py::arg("dense_masks"), py::arg("dense_mask_queries"),
          py::arg("n_queries"), py::arg("n_docs"));
    m.def("cluster_legals_preordered_packed", &cluster_legals_preordered_packed_gpu_out_py,
          py::arg("mask_packed_buffer"), py::arg("n_queries"), py::arg("n_docs"),
          py::arg("cluster_offsets"), py::arg("cluster_counts"), py::arg("n_list"));
    m.def("get_top_rankings", &get_top_rankings_out_py,
          py::arg("clustered_legals"), py::arg("cluster_dist"),
          py::arg("n_queries"), py::arg("n_list"), py::arg("n_probe"),
          py::arg("beta"), py::arg("eps") = 1e-9f);
}
