#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <future>
#include <thread>
#include <type_traits>
#include <memory>
#include <stdexcept>
#include <variant>
#include <functional>
#include <vector>

// cluster_legals_gpu_kernel implements the cuda logic that pybind interfaces with.

inline void checkGpuStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

inline int GpuAlignment = 16;

inline size_t round(size_t number, int roundedNumber) {
    number = (number + roundedNumber - 1);
    number /= roundedNumber;
    number *= roundedNumber;
    return number;
}
inline size_t roundToAlignment(size_t number) {
    return round(number, GpuAlignment);
}

// Shared handle to a cuda stream. copies refer to the same stream.
struct StreamCapsule {
    std::shared_ptr<std::remove_pointer_t<cudaStream_t>> m_stream;

    // Creates a new stream.
    StreamCapsule();

    cudaStream_t get() const { return m_stream.get(); }
    void sync() const;
    bool operator==(const StreamCapsule& other) const { return m_stream == other.m_stream; }
};

class TimeStampData;

// Represents a block of data held up before a cuda event.
struct PromiseData {
protected:
    void* m_data{ nullptr };
    size_t m_size{ 0 };

    // Event that marks the completion of the promise
    std::shared_ptr<std::remove_pointer_t<cudaEvent_t>> m_endEvent;
    std::shared_ptr<std::remove_pointer_t<cudaEvent_t>> m_startEvent;
    StreamCapsule m_stream;

    PromiseData(size_t size, StreamCapsule stream);
    ~PromiseData();

public:
    PromiseData(const PromiseData&) = delete;
    PromiseData& operator=(const PromiseData&) = delete;

    // Sets an event at the end of the stream.
    void finalizePromise();

    // Getters and setters
    StreamCapsule getStream() const;
    TimeStampData getStartTimeStamp() const;
    TimeStampData getEndTimeStamp() const;
    size_t getSize() const;
    const void* getData() const;
    void* getData();

    // Forces event to be synced
    void sync() const;

    void makeStreamWait(StreamCapsule stream) const;
    // To avoid having raw pointer type in python, 
    // integer is use as handle on Cuda stream.
    void makeStreamWait(int64_t stream) const;
};

// Owns pinned Cpu data that may be in transit from the Gpu.
class ReadbackPromiseData : public PromiseData {
public:
    ReadbackPromiseData(size_t size, StreamCapsule stream = StreamCapsule());
    ~ReadbackPromiseData();

    // Caller checks capacity and retains d_data until the copy completes.
    void uploadGpuDataAsync(const void* d_data, size_t size, size_t offset);
};

// Represents a block of data to be completed within the Gpu
class GpuPromiseData : public PromiseData {
public:
    GpuPromiseData(size_t size, StreamCapsule stream = StreamCapsule());
    ~GpuPromiseData();

    void uploadValueAsync(int value, size_t size, size_t offset);
    // Up to caller to retain data within h_data until data has been fully copied in
    void uploadCpuDataAsync(const void* h_data, size_t size, size_t offset);
    // Returns a completed CPU copy.
    std::shared_ptr<const ReadbackPromiseData> collectRawCpuCopy() const;
    // Copies a portion of the gpu data within bounds to cpu
    std::shared_ptr<const ReadbackPromiseData> copyToCpu(const void* d_gpuData, size_t size) const;
};

// Owns an event and handles 
// The thing being time stamped may not have even happened yet 
class TimeStampData {
private:
    // Event being time stamped
    std::shared_ptr<std::remove_pointer_t<cudaEvent_t>> m_event{ nullptr };

public:
    TimeStampData(std::shared_ptr<std::remove_pointer_t<cudaEvent_t>> event);

    // Synchronizes Gpu such that time stamped event actually happens
    void sync() const;

    // The time in milliseconds a Cuda event happened from some unspecified base
    float getTimeStamp() const;
};

std::shared_ptr<GpuPromiseData> predicate_results_to_bitmap_gpu_impl (
    std::vector<const PromiseData*> blockingPromises,
    const uint32_t* d_postingLists,
    const int32_t* d_postingSizes,
    const int32_t* d_postingQueries,
    const uint8_t* d_denseMasks,
    const int64_t* h_denseMaskQueries,
    int32_t nQueries,
    int32_t nDocs,
    int32_t postingCount,
    int32_t denseMaskCount,
    int64_t postingEntries);


std::shared_ptr<GpuPromiseData> cluster_legals_preordered_packed_gpu_impl (
    std::vector<const PromiseData*> blockingPromises,
    const uint8_t* d_maskPackedBuffer,
    int32_t nQueries,
    int32_t nDocs,
    const int32_t* d_clusterOffsets,
    const int32_t* d_clusterCounts,
    int32_t nList);

std::shared_ptr<const ReadbackPromiseData> get_top_rankings_gpu_impl (
    std::vector<const PromiseData*> blockingPromises,
    const float* d_clusteredLegals,
    const float* d_clusterDist,
    int32_t nQueries,
    int32_t nList,
    int32_t nProbe,
    float beta,
    float eps);
