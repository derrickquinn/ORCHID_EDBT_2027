#include "cluser_legals_gpu_kernel.h"

#include <cub/cub.cuh>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;
#include <thrust/system/cuda/execution_policy.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

// Used to synchronize timestamps
static std::shared_ptr<std::remove_pointer_t<cudaEvent_t>> g_earliestEvent;

void StreamCapsule::sync() const {
    checkGpuStatus(cudaStreamSynchronize(get()));
}

StreamCapsule::StreamCapsule() {
    cudaStream_t stream = nullptr;
    checkGpuStatus(cudaStreamCreate(&stream));
    m_stream.reset(stream, [](cudaStream_t s) { cudaStreamDestroy(s); });
}

// Funnels a number of promises into a single Gpu stream.
static StreamCapsule funnelStreams(const std::vector<const PromiseData*>& promises) {
    if (promises.empty()) {
        return StreamCapsule();
    }
    StreamCapsule extendedStream = promises.front()->getStream();
    for (int i = 1 ; i < promises.size() ; i++) {
        promises[i]->makeStreamWait(extendedStream);
    }
    return extendedStream;
}
PromiseData::PromiseData(size_t size, StreamCapsule stream)
    : m_size(size), m_stream(std::move(stream)) {
    if (m_size == 0) {
        throw std::runtime_error("Promise can't hold no information whatsoever");
    }

    cudaEvent_t event = nullptr;
    checkGpuStatus(cudaEventCreate(&event));
    m_startEvent.reset(event, [](cudaEvent_t e) { cudaEventDestroy(e); });

    checkGpuStatus(cudaEventRecord(event, m_stream.get()));

    if (g_earliestEvent.get() == nullptr) {
        g_earliestEvent = m_startEvent;
        checkGpuStatus(cudaEventSynchronize(g_earliestEvent.get()));
    }
}

PromiseData::~PromiseData() = default;

ReadbackPromiseData::ReadbackPromiseData(size_t size, StreamCapsule stream)
    : PromiseData(size, std::move(stream)) {
    if (m_size != 0) {
        checkGpuStatus(cudaMallocHost(&m_data, roundToAlignment(m_size)));
    }
}

ReadbackPromiseData::~ReadbackPromiseData() {
    if (m_data) {
        cudaFreeHost(m_data);
    }
}

void ReadbackPromiseData::uploadGpuDataAsync(const void* d_data, size_t size, size_t offset) {
    if (size != 0) {
        void* h_destination = static_cast<uint8_t*>(m_data) + offset;
        checkGpuStatus(cudaMemcpyAsync(h_destination, d_data, size,
                                       cudaMemcpyDeviceToHost, m_stream.get()));
    }
}

GpuPromiseData::GpuPromiseData(size_t size, StreamCapsule stream)
    : PromiseData(size, std::move(stream)) {
    checkGpuStatus(cudaMallocAsync(&m_data, roundToAlignment(m_size), m_stream.get()));

}
GpuPromiseData::~GpuPromiseData() {
    cudaFreeAsync(m_data, m_stream.get());
}
void PromiseData::finalizePromise() {
    if (m_endEvent) {
        throw std::runtime_error("Promise can't be finalized multiple times");
    }

    cudaEvent_t event = nullptr;
    cudaEventCreate(&event);
    m_endEvent.reset(event, [](cudaEvent_t e) { cudaEventDestroy(e); });
    cudaEventRecord(event, m_stream.get());
}
const void* PromiseData::getData() const {
    return m_data;
}
void* PromiseData::getData() {
    return m_data;
}
StreamCapsule PromiseData::getStream() const {
    return m_stream;
}
TimeStampData PromiseData::getStartTimeStamp() const {
    return TimeStampData(m_startEvent);
}
TimeStampData PromiseData::getEndTimeStamp() const {
    if (!m_endEvent) {
        throw std::runtime_error("Promise must be finalized before reading its end time stamp");
    }
    return TimeStampData(m_endEvent);
}
size_t PromiseData::getSize() const {
    return m_size;
}
void GpuPromiseData::uploadValueAsync(int value, size_t size, size_t offset) {
    if (size != 0) {
        void* d_destination = static_cast<uint8_t*>(m_data) + offset;
        checkGpuStatus(cudaMemsetAsync(d_destination, value, size, m_stream.get()));
    }
}
void GpuPromiseData::uploadCpuDataAsync(const void* h_data, size_t size, size_t offset) {
    if (size != 0) {
        void* d_destination = static_cast<uint8_t*>(m_data) + offset;
        checkGpuStatus(cudaMemcpyAsync(d_destination, h_data, size,
                                       cudaMemcpyHostToDevice, m_stream.get()));
    }
}

void PromiseData::sync() const {
    if (m_endEvent) {
        checkGpuStatus(cudaEventSynchronize(m_endEvent.get()));
    } else {
        checkGpuStatus(cudaStreamSynchronize(m_stream.get()));
    }
}

void PromiseData::makeStreamWait(StreamCapsule stream) const {
    if (!m_endEvent) {
        throw std::runtime_error("Promise must be finalized before another stream can wait on it");
    }
    checkGpuStatus(cudaStreamWaitEvent(
        stream.get(), m_endEvent.get(), 0));
}
void PromiseData::makeStreamWait(int64_t stream) const {
    if (!m_endEvent) {
        throw std::runtime_error("Promise must be finalized before another stream can wait on it");
    }
    checkGpuStatus(cudaStreamWaitEvent(
        reinterpret_cast<cudaStream_t>(stream), m_endEvent.get(), 0));
}

std::shared_ptr<const ReadbackPromiseData> GpuPromiseData::collectRawCpuCopy() const {
    std::shared_ptr<ReadbackPromiseData> result = std::make_shared<ReadbackPromiseData>(m_size, m_stream);
    result->uploadGpuDataAsync(m_data, m_size, 0);
    result->finalizePromise();
    return result;
}


std::shared_ptr<const ReadbackPromiseData> GpuPromiseData::copyToCpu(const void* d_gpuData, size_t size) const {
    std::shared_ptr<ReadbackPromiseData> result = std::make_shared<ReadbackPromiseData>(size, m_stream);
    result->uploadGpuDataAsync(d_gpuData, size, 0);
    result->finalizePromise();
    return result;
}

    // std::shared_ptr<std::remove_pointer_t<cudaEvent_t>> m_event;

TimeStampData::TimeStampData(std::shared_ptr<std::remove_pointer_t<cudaEvent_t>> event)
    : m_event(std::move(event)) {}

void TimeStampData::sync() const {
    checkGpuStatus(cudaEventSynchronize(m_event.get()));
}

float TimeStampData::getTimeStamp() const {
    checkGpuStatus(cudaEventSynchronize(m_event.get()));
    float ms;
    checkGpuStatus(cudaEventElapsedTime(&ms, g_earliestEvent.get(), m_event.get()));
    return ms;
}


// Assigns each thread a chunk of one posting list.
__global__ void postingListKernel (
    const uint32_t* d_postingLists,
    const int64_t* d_postingThreads,
    const int64_t* d_postingSizes,
    const int32_t* d_postingQueries,
    uint8_t* d_outMask,
    int32_t bytesPerQuery,
    int32_t threadWidth,
    int32_t postingCount) {
    extern __shared__ int64_t d_sharedMem[];
    
    // Sets up shared memory
    auto block = cg::this_thread_block();
    int64_t* d_postingSizesShared = d_sharedMem;
    int64_t* d_postingThreadsShared = d_postingSizesShared + postingCount;
    cg::memcpy_async(block, d_postingThreadsShared, d_postingThreads, postingCount * sizeof(int64_t));
    cg::memcpy_async(block, d_postingSizesShared, d_postingSizes, postingCount * sizeof(int64_t));

    // Gets the query the thread will be operating in
    const int64_t threadGlobalIdx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    cg::wait_prior<1>(block);
    const int32_t queryIdx = cub::UpperBound(d_postingThreadsShared, postingCount, threadGlobalIdx);

    // Finish the collective before idle threads exit.
    cg::wait(block);
    if (postingCount == queryIdx) {
        return;
    }

    // Gets the posting list bounds of the thread
    const int64_t prevPostingSizes = queryIdx == 0 ? 0 : d_postingSizesShared[queryIdx - 1];
    const int64_t prevPostingThreads = queryIdx == 0 ? 0 : d_postingThreadsShared[queryIdx - 1];
    const int64_t queryStartIdx = prevPostingSizes + (threadGlobalIdx - prevPostingThreads) * threadWidth;
    const int64_t queryEndIdx = cuda::std::min(queryStartIdx + threadWidth, d_postingSizesShared[queryIdx]);

    const int64_t maskBegin = static_cast<int64_t>(bytesPerQuery) * d_postingQueries[queryIdx];

    // Sets the bits for this thread's posting range.
    for (int64_t i = queryStartIdx ; i < queryEndIdx; i++) {
        const uint32_t id = d_postingLists[i];
        const int64_t byteIdx = maskBegin + (id / 8);
        const unsigned int bitIdx = static_cast<unsigned int>((byteIdx & 3) * 8 + (id & 7));
        atomicOr(reinterpret_cast<unsigned int*>(d_outMask) + (byteIdx / sizeof(unsigned int)), 1u << bitIdx);
    }
}

__device__ static inline uint32_t popcount_bytes(const uint8_t* data, int32_t len) {
  uint64_t total = 0;
  int32_t i = 0;
  for (; i + 8 <= len; i += 8) {
    uint64_t v;
    cuda::std::memcpy(&v, data + i, sizeof(uint64_t));
    total += static_cast<uint64_t>(__popcll(v));
  }
  for (; i < len; ++i) {
    total += static_cast<uint64_t>(__popcll(data[i]));
  }
  return static_cast<uint32_t>(total);
}

// TODO: likely horribly optimized
// Fix later
__global__ void clusterLegalsKernel (
    float* out,
    const uint8_t* d_maskPackedBuffer,
    int32_t nQueries,
    int32_t bytesPerEntry,
    const int32_t* d_clusterOffsets,
    const int32_t* d_clusterCounts,
    int32_t nList,
    int32_t blockCount) {

    const int32_t blockQueries = ((nQueries + blockCount - 1) / blockCount);
    const int32_t blockStart = blockIdx.x * blockQueries;
    const int32_t blockEnd = cuda::std::min(blockStart + blockQueries, nQueries);
     
    const int32_t threadWidth = (nList + blockDim.x - 1)/ blockDim.x;
    const int32_t threadStart = threadIdx.x * threadWidth;
    const int32_t threadEnd = cuda::std::min(threadStart + threadWidth, nList);
    for (int32_t q = blockStart; q < blockEnd ; ++q) {
        const uint8_t* row =
            d_maskPackedBuffer + static_cast<int64_t>(q) * bytesPerEntry;
        float* out_row = out + static_cast<int64_t>(q) * nList;
        for (int32_t c = threadStart; c < threadEnd ; ++c) {
            int32_t start = d_clusterOffsets[c];
            int32_t end = d_clusterOffsets[c + 1];
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
                sum += static_cast<uint32_t>(__popc(row[s_byte] & mask));
            } else {
                if (s_bit != 0) {
                    uint8_t mask = static_cast<uint8_t>(0xFFu << s_bit);
                    sum += static_cast<uint32_t>(__popc(row[s_byte] & mask));
                    s_byte += 1;
                }

                if (s_byte < e_byte) {
                    sum += popcount_bytes(row + s_byte, e_byte - s_byte);
                }

                if (e_bit != 0) {
                    uint8_t mask = static_cast<uint8_t>((1u << e_bit) - 1u);
                    sum += static_cast<uint32_t>(__popc(row[e_byte] & mask));
                }
            }

            int32_t cnt = d_clusterCounts[c];
            out_row[c] = cnt > 0 ? static_cast<float>(sum) / static_cast<float>(cnt)
                                : 0.0f;
        }
    }
}
struct ThreadWidthDivision {
    int32_t threadWidth{1};

    ThreadWidthDivision(int32_t width) : threadWidth(width) {}

    __host__ __device__
    int64_t operator()(const int32_t& x) const {
        return x / threadWidth + (x % threadWidth != 0);
    }
};

std::shared_ptr<GpuPromiseData> predicate_results_to_bitmap_gpu_impl(
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
    int64_t postingEntries) {
    const int32_t bytesPerQuery = nDocs / 8 + (nDocs % 8 != 0);
    const size_t totalSize = static_cast<size_t>(nQueries) * bytesPerQuery;
    auto extendedPromise = funnelStreams(blockingPromises);
    auto ret = std::make_shared<GpuPromiseData>(totalSize, extendedPromise);
    const cudaStream_t stream = extendedPromise.get();
    
    if (ret->getData()) {
        ret->uploadValueAsync(0, ret->getSize(), 0);
    }

    // Streams posting lists into logic
    int64_t* d_postingMetadata = nullptr;
    if (postingCount) {
        const size_t metadataBytes = static_cast<size_t>(postingCount) * 2 * sizeof(int64_t);
        checkGpuStatus(cudaMallocAsync(
            reinterpret_cast<void**>(&d_postingMetadata), metadataBytes, stream));
        int64_t* d_threadCounts = d_postingMetadata;
        int64_t* d_postingOffsets = d_postingMetadata + postingCount;
        const int32_t threadWidth = 128;
        const auto policy = thrust::cuda::par_nosync.on(stream);
        thrust::transform(
            policy, d_postingSizes, d_postingSizes + postingCount,
            d_threadCounts, ThreadWidthDivision{threadWidth});
        thrust::inclusive_scan(policy, d_threadCounts, d_threadCounts + postingCount, d_threadCounts);
        // Preserve input sizes and accumulate offsets without 32-bit overflow.
        thrust::transform(
            policy, d_postingSizes, d_postingSizes + postingCount,
            d_postingOffsets, ThreadWidthDivision{1});
        thrust::inclusive_scan(policy, d_postingOffsets, d_postingOffsets + postingCount, d_postingOffsets);

        // Sum ceil(size / threadWidth) <= postingCount + floor(entries / threadWidth).
        // The uploaded span provides this bound without copying a scan result to the CPU.
        const int64_t totalThreads = postingCount + postingEntries / threadWidth;
        const int blocks = static_cast<int>((totalThreads + 255) / 256);

        postingListKernel<<<blocks, 256, metadataBytes, stream>>>(
            d_postingLists, d_threadCounts, d_postingOffsets, d_postingQueries,
            static_cast<uint8_t*>(ret->getData()), bytesPerQuery,
            threadWidth, postingCount);

        checkGpuStatus(cudaGetLastError());
        checkGpuStatus(cudaFreeAsync(d_postingMetadata, stream));
        d_postingMetadata = nullptr;
    }

    // Run after posting atomics: adjacent rows may share an atomic word.
    for (int32_t i = 0; i < denseMaskCount; ++i) {
        checkGpuStatus(cudaMemcpyAsync(
            static_cast<uint8_t*>(ret->getData()) + h_denseMaskQueries[i] * bytesPerQuery,
            d_denseMasks + static_cast<int64_t>(i) * bytesPerQuery,
            bytesPerQuery, cudaMemcpyDeviceToDevice, stream));
    }
    ret->finalizePromise();
    return ret;
}


std::shared_ptr<GpuPromiseData> cluster_legals_preordered_packed_gpu_impl (
    std::vector<const PromiseData*> blockingPromises,
    const uint8_t* d_maskPackedBuffer,
    int32_t nQueries,
    int32_t nDocs,
    const int32_t* d_clusterOffsets,
    const int32_t* d_clusterCounts,
    int32_t nList) {
    // Matches the row stride of predicate_results_to_bitmap_gpu_impl.
    const int32_t bytesPerEntry = nDocs / 8 + (nDocs % 8 != 0);
    const size_t totalSize = static_cast<size_t>(nQueries) * nList * sizeof(float);
    auto extendedPromise = funnelStreams(blockingPromises);
    auto ret = std::make_shared<GpuPromiseData>(totalSize, extendedPromise);
    const cudaStream_t stream = extendedPromise.get();

    // One block per query, with the block's threads splitting the clusters.
    clusterLegalsKernel<<<nQueries, 256, 0, stream>>>(
        static_cast<float*>(ret->getData()), d_maskPackedBuffer, nQueries, bytesPerEntry,
        d_clusterOffsets, d_clusterCounts, nList, nQueries);
    checkGpuStatus(cudaGetLastError());

    ret->finalizePromise();
    return ret;
}


// TODO: stub, not implemented yet.
// Rank clusters by distance + beta * log(legality + eps), returning nProbe IDs per query.
__global__ void logScaleKernel (
    int64_t* out,
    const float* d_clusteredLegals,
    const float* d_clusterDist,
    int32_t nQueries,
    int32_t nList,
    int32_t nProbe,
    float beta,
    float eps) {
}

std::shared_ptr<const ReadbackPromiseData> get_top_rankings_gpu_impl (
    std::vector<const PromiseData*> blockingPromises,
    const float* d_clusteredLegals,
    const float* d_clusterDist,
    int32_t nQueries,
    int32_t nList,
    int32_t nProbe,
    float beta,
    float eps) {
    const int64_t count = static_cast<int64_t>(nQueries) * nProbe;
    const size_t totalSize = static_cast<size_t>(count) * sizeof(int64_t);
    auto extendedPromise = funnelStreams(blockingPromises);
    auto ret = std::make_shared<GpuPromiseData>(totalSize, extendedPromise);
    const cudaStream_t stream = extendedPromise.get();

    if (count > 0) {
        constexpr int32_t threadsPerBlock = 256;
        const int64_t blocks = (count + threadsPerBlock - 1) / threadsPerBlock;
        logScaleKernel<<<static_cast<unsigned int>(blocks), threadsPerBlock, 0, stream>>>(
            static_cast<int64_t*>(ret->getData()), d_clusteredLegals, d_clusterDist,
            nQueries, nList, nProbe, beta, eps);
        checkGpuStatus(cudaGetLastError());
    }

    ret->finalizePromise();
    return ret->collectRawCpuCopy();
}
