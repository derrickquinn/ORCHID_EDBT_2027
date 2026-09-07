#include <faiss/IndexHNSW.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

std::vector<float> read_fvecs(const std::string &path, std::size_t count,
                              int &dimension) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("could not open " + path);
  }

  std::int32_t first_dimension = 0;
  input.read(reinterpret_cast<char *>(&first_dimension), sizeof(first_dimension));
  if (!input || first_dimension <= 0) {
    throw std::runtime_error("invalid fvecs header");
  }
  dimension = first_dimension;
  input.seekg(0);

  std::vector<float> vectors(count * static_cast<std::size_t>(dimension));
  for (std::size_t row = 0; row < count; ++row) {
    std::int32_t row_dimension = 0;
    input.read(reinterpret_cast<char *>(&row_dimension), sizeof(row_dimension));
    if (!input || row_dimension != dimension) {
      throw std::runtime_error("fvecs ended early or changed dimension");
    }
    input.read(reinterpret_cast<char *>(vectors.data() + row * dimension),
               static_cast<std::streamsize>(dimension * sizeof(float)));
    if (!input) {
      throw std::runtime_error("fvecs ended before the requested row count");
    }
  }
  return vectors;
}

void normalize_rows(std::vector<float> &vectors, int dimension, int threads) {
  const std::int64_t count =
      static_cast<std::int64_t>(vectors.size() / dimension);
#pragma omp parallel for schedule(static) num_threads(threads)
  for (std::int64_t row = 0; row < count; ++row) {
    float *values = vectors.data() + row * dimension;
    double squared_norm = 0.0;
    for (int column = 0; column < dimension; ++column) {
      squared_norm += static_cast<double>(values[column]) * values[column];
    }
    if (squared_norm == 0.0) {
      continue;
    }
    const float inverse_norm =
        static_cast<float>(1.0 / std::sqrt(squared_norm));
    for (int column = 0; column < dimension; ++column) {
      values[column] *= inverse_norm;
    }
  }
}

int parse_positive(const char *text, const char *name) {
  const int value = std::stoi(text);
  if (value <= 0) {
    throw std::invalid_argument(std::string(name) + " must be positive");
  }
  return value;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 6) {
      std::cerr << "usage: " << argv[0]
                << " BASE.fvecs COUNT M EF_CONSTRUCTION THREADS\n";
      return 2;
    }

    const std::string path = argv[1];
    const int count = parse_positive(argv[2], "count");
    const int m = parse_positive(argv[3], "M");
    const int ef_construction = parse_positive(argv[4], "efConstruction");
    const int threads = parse_positive(argv[5], "threads");
    omp_set_num_threads(threads);

    int dimension = 0;
    auto start = Clock::now();
    std::vector<float> vectors = read_fvecs(path, count, dimension);
    std::cout << "load_seconds=" << seconds_since(start) << '\n';

    start = Clock::now();
    normalize_rows(vectors, dimension, threads);
    std::cout << "normalize_seconds=" << seconds_since(start) << '\n';

    faiss::IndexHNSWFlat index(dimension, m, faiss::METRIC_L2);
    index.hnsw.efConstruction = ef_construction;
    start = Clock::now();
    index.add(count, vectors.data());
    const double build_seconds = seconds_since(start);

    std::cout << "count=" << count << '\n'
              << "dimension=" << dimension << '\n'
              << "M=" << m << '\n'
              << "ef_construction=" << ef_construction << '\n'
              << "threads=" << threads << '\n'
              << "build_seconds=" << build_seconds << '\n'
              << "ntotal=" << index.ntotal << '\n';
    return index.ntotal == count ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
