#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <unordered_set>
#include <vector>
#include <chrono>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/IDSelector2D.h>
#include <cinttypes>    
#include <faiss/Index.h>

static void fill_random(float* x, size_t n, size_t d, unsigned seed = 123) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (size_t i = 0; i < n * d; ++i) x[i] = nd(rng);
}

int main() {
    using namespace faiss;

    const size_t d = 32;
    const size_t nb = 2000;
    const size_t nq = 3;
    const size_t k = 5;
    const size_t nlist = 64;
    const size_t nprobe = 8;

    printf("=== Testing 2D Selector Implementation ===\n");

    // 1) Data setup
    std::vector<float> xb(nb * d);
    std::vector<float> xq(nq * d);
    fill_random(xb.data(), nb, d, 123);
    fill_random(xq.data(), nq, d, 321);

    // 2) Build IVF index
    IndexFlatL2 quantizer(d);
    IndexIVFFlat index(&quantizer, d, nlist, METRIC_L2);

    index.train(nb, xb.data());
    std::vector<idx_t> ids(nb);
    for (idx_t i = 0; i < (idx_t)nb; ++i) ids[i] = i;
    index.add_with_ids(nb, xb.data(), ids.data());
    index.nprobe = nprobe;

    // 3) Test IDSelector2DHashed - Different allow sets per query
    printf("\n--- Testing IDSelector2DHashed ---\n");
    
    std::vector<std::unordered_set<idx_t>> allow_sets(nq);
    
    // Query 0: only even IDs
    for (idx_t id = 0; id < (idx_t)nb; ++id) {
        if (id % 2 == 0) allow_sets[0].insert(id);
    }
    
    // Query 1: only odd IDs  
    for (idx_t id = 0; id < (idx_t)nb; ++id) {
        if (id % 2 == 1) allow_sets[1].insert(id);
    }
    
    // Query 2: only IDs divisible by 3
    for (idx_t id = 0; id < (idx_t)nb; ++id) {
        if (id % 3 == 0) allow_sets[2].insert(id);
    }

    IDSelector2DHashed sel2d_hashed(std::move(allow_sets));

    // REMOVED: Custom struct definition - use standard parameters
    SearchParametersIVF params;
    params.nprobe = nprobe;
    params.sel2d = &sel2d_hashed;

    std::vector<float> D(nq * k);
    std::vector<idx_t> I(nq * k);

    // Search with 2D selector
    index.search(nq, xq.data(), k, D.data(), I.data(), &params);

    // 4) Validate results
    auto validate_results = [&](const char* test_name) {
        printf("Results for %s:\n", test_name);
        bool all_valid = true;
        
        for (size_t q = 0; q < nq; ++q) {
            printf("Query %zu: ", q);
            bool query_valid = true;
            
            for (size_t j = 0; j < k; ++j) {
                idx_t result_id = I[q * k + j];
                float distance = D[q * k + j];
                
                if (result_id >= 0) {
                    bool should_be_allowed = false;
                    if (q == 0) should_be_allowed = (result_id % 2 == 0);
                    else if (q == 1) should_be_allowed = (result_id % 2 == 1);  
                    else if (q == 2) should_be_allowed = (result_id % 3 == 0);
                    
                    if (!should_be_allowed) {
                        printf("INVALID(id=%" PRId64 ") ", result_id);
                        query_valid = false;
                    } else {
                        printf("(%.3f,%" PRId64 ") ", distance, result_id);
                    }
                } else {
                    printf("(%.3f,%" PRId64 ") ", distance, result_id);
                }
            }
            
            if (!query_valid) all_valid = false;
            printf("%s\n", query_valid ? "✓" : "✗");
        }
        
        return all_valid;
    };

    bool hashed_valid = validate_results("IDSelector2DHashed");

    // 5) Test IDSelector2DBitmap
    printf("\n--- Testing IDSelector2DBitmap ---\n");
    
    // Create bitmap data
    size_t bytes_per_bitmap = (nb + 7) / 8; // ceil(nb / 8)
    std::vector<uint8_t> bitmap_data(nq * bytes_per_bitmap, 0);
    
    // Set bits for allowed IDs
    for (size_t q = 0; q < nq; ++q) {
        uint8_t* bm = bitmap_data.data() + q * bytes_per_bitmap;
        
        for (idx_t id = 0; id < (idx_t)nb; ++id) {
            bool allow = false;
            if (q == 0) allow = (id % 2 == 0);       // even
            else if (q == 1) allow = (id % 2 == 1);  // odd
            else if (q == 2) allow = (id % 3 == 0);  // divisible by 3
            
            if (allow) {
                bm[id >> 3] |= (1 << (id & 7));
            }
        }
    }

    IDSelector2DBitmap sel2d_bitmap;
    sel2d_bitmap.base = bitmap_data.data();
    sel2d_bitmap.bytes_per_bitmap = bytes_per_bitmap;
    sel2d_bitmap.N = nb;
    sel2d_bitmap.nq = nq;

    // Search with bitmap selector - REUSE same params object
    params.sel2d = &sel2d_bitmap;
    std::fill(D.begin(), D.end(), 0.0f);
    std::fill(I.begin(), I.end(), -1);
    
    index.search(nq, xq.data(), k, D.data(), I.data(), &params);
    
    bool bitmap_valid = validate_results("IDSelector2DBitmap");

    // 6) Test adapter class - RECREATE allow_sets since they were moved
    printf("\n--- Testing IDSelectorFrom2DForQ ---\n");
    
    // Recreate the allow sets for testing the adapter
    std::vector<std::unordered_set<idx_t>> test_allow_sets(nq);
    for (idx_t id = 0; id < (idx_t)nb; ++id) {
        if (id % 2 == 0) test_allow_sets[0].insert(id);
        if (id % 2 == 1) test_allow_sets[1].insert(id);
        if (id % 3 == 0) test_allow_sets[2].insert(id);
    }
    IDSelector2DHashed test_sel2d(std::move(test_allow_sets));
    
    for (size_t q = 0; q < nq; ++q) {
        IDSelectorFrom2DForQ adapter(&test_sel2d, q);
        
        // Test some IDs
        printf("Query %zu adapter: ", q);
        int test_count = 0;
        for (idx_t test_id = 0; test_id < 10 && test_id < nb; ++test_id) {
            bool expected = false;
            if (q == 0) expected = (test_id % 2 == 0);
            else if (q == 1) expected = (test_id % 2 == 1);
            else if (q == 2) expected = (test_id % 3 == 0);
            
            bool actual = adapter.is_member(test_id);
            if (actual == expected) {
                test_count++;
            } else {
                printf("FAIL(id=%" PRId64 " expected=%d actual=%d) ", 
                       test_id, expected, actual);
            }
        }
        printf("%d/10 tests passed\n", test_count);
    }

    // 7) Performance comparison test
    printf("\n--- Performance Comparison ---\n");
    
    auto time_search = [&](const char* name, const IDSelector2D* sel) {
        params.sel2d = sel;
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int trial = 0; trial < 10; ++trial) {
            index.search(nq, xq.data(), k, D.data(), I.data(), &params);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        printf("%s: %.2f μs/search\n", name, duration.count() / 10.0);
    };

    time_search("No selector", nullptr);
    time_search("Hashed 2D", &sel2d_hashed);
    time_search("Bitmap 2D", &sel2d_bitmap);

    // 8) Final results
    printf("\n=== Test Results ===\n");
    printf("IDSelector2DHashed: %s\n", hashed_valid ? "PASSED" : "FAILED");
    printf("IDSelector2DBitmap: %s\n", bitmap_valid ? "PASSED" : "FAILED");
    
    if (hashed_valid && bitmap_valid) {
        printf("All 2D selector tests PASSED! ✓\n");
        printf("Your implementation successfully enables per-query filtering in batched searches.\n");
        return 0;
    } else {
        printf("Some tests FAILED! ✗\n"); 
        return 1;
    }
}