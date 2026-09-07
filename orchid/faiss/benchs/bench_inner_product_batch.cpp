#include <faiss/utils/distances.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

float benchmark_scalar(
        const float* query,
        const float* docs,
        size_t num_docs,
        size_t dim,
        size_t iterations,
        volatile float& sink) {
    float result_us = 0.0f;
    auto start = Clock::now();
    for (size_t it = 0; it < iterations; ++it) {
        float acc = 0.0f;
        for (size_t j = 0; j < num_docs; ++j) {
            acc += faiss::fvec_inner_product(query, docs + j * dim, dim);
        }
        sink += acc;
    }
    auto end = Clock::now();
    result_us = std::chrono::duration<float, std::micro>(end - start).count();
    return result_us / static_cast<float>(iterations);
}

float benchmark_batch4(
        const float* query,
        const float* docs,
        size_t num_docs,
        size_t dim,
        size_t iterations,
        volatile float& sink) {
    float result_us = 0.0f;
    auto start = Clock::now();
    for (size_t it = 0; it < iterations; ++it) {
        float acc = 0.0f;
        size_t j = 0;
        for (; j + 3 < num_docs; j += 4) {
            float d0 = 0.0f;
            float d1 = 0.0f;
            float d2 = 0.0f;
            float d3 = 0.0f;
            faiss::fvec_inner_product_batch_4(
                    query,
                    docs + (j + 0) * dim,
                    docs + (j + 1) * dim,
                    docs + (j + 2) * dim,
                    docs + (j + 3) * dim,
                    dim,
                    d0,
                    d1,
                    d2,
                    d3);
            acc += d0 + d1 + d2 + d3;
        }
        for (; j < num_docs; ++j) {
            acc += faiss::fvec_inner_product(query, docs + j * dim, dim);
        }
        sink += acc;
    }
    auto end = Clock::now();
    result_us = std::chrono::duration<float, std::micro>(end - start).count();
    return result_us / static_cast<float>(iterations);
}

float benchmark_batch8(
        const float* query,
        const float* docs,
        size_t num_docs,
        size_t dim,
        size_t iterations,
        volatile float& sink) {
    float result_us = 0.0f;
    std::array<const float*, 8> vec_ptrs{};
    std::array<float, 8> block{};

    auto start = Clock::now();
    for (size_t it = 0; it < iterations; ++it) {
        float acc = 0.0f;
        size_t j = 0;
        for (; j + 7 < num_docs; j += 8) {
            for (size_t k = 0; k < 8; ++k) {
                vec_ptrs[k] = docs + (j + k) * dim;
            }
            faiss::fvec_inner_product_batch_8(query, vec_ptrs.data(), dim, block.data());
            for (float v : block) {
                acc += v;
            }
        }
        for (; j < num_docs; ++j) {
            acc += faiss::fvec_inner_product(query, docs + j * dim, dim);
        }
        sink += acc;
    }
    auto end = Clock::now();
    result_us = std::chrono::duration<float, std::micro>(end - start).count();
    return result_us / static_cast<float>(iterations);
}

size_t choose_iteration_count(size_t num_docs, size_t dim) {
    constexpr size_t target_ops = 5'000'000'000;
    size_t denom = num_docs * dim;
    if (denom == 0) {
        return 1;
    }
    size_t iters = target_ops / denom;
    if (iters < 10) {
        iters = 10;
    }
    return iters;
}

} // namespace

int main() {
    std::vector<size_t> doc_counts = {64, 256, 1024};
    std::vector<size_t> dimensions = {128, 256, 512};

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    volatile float sink = 0.0f;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "method,docs,dim,iters,us_per_iter" << std::endl;

    for (size_t num_docs : doc_counts) {
        for (size_t dim : dimensions) {
            std::vector<float> query(dim);
            for (float& v : query) {
                v = dist(rng);
            }

            std::vector<float> docs(num_docs * dim);
            for (float& v : docs) {
                v = dist(rng);
            }

            size_t iterations = choose_iteration_count(num_docs, dim);

            float scalar_us = benchmark_scalar(
                    query.data(), docs.data(), num_docs, dim, iterations, sink);
            std::cout << "scalar," << num_docs << ',' << dim << ',' << iterations
                      << ',' << scalar_us << std::endl;

            float batch4_us = benchmark_batch4(
                    query.data(), docs.data(), num_docs, dim, iterations, sink);
            std::cout << "batch4," << num_docs << ',' << dim << ',' << iterations
                      << ',' << batch4_us << std::endl;

            float batch8_us = benchmark_batch8(
                    query.data(), docs.data(), num_docs, dim, iterations, sink);
            std::cout << "batch8," << num_docs << ',' << dim << ',' << iterations
                      << ',' << batch8_us << std::endl;
        }
    }

    std::cout << "sink," << sink << std::endl;
    return 0;
}
