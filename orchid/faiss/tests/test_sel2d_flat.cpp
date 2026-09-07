#include <algorithm>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <unordered_set>
#include <vector>

#include <faiss/Index.h>                  
#include <faiss/IndexFlat.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/IDSelector2D.h>

static void fill_random(float* x, size_t n, size_t d, unsigned seed = 123) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (size_t i = 0; i < n * d; ++i) x[i] = nd(rng);
}

int main() {
    using namespace faiss;

    const size_t d  = 32;
    const size_t nb = 5000;
    const size_t nq = 3;
    const size_t k  = 5;

    printf("=== Testing 2D Selector on IndexFlat ===\n");

    std::vector<float> xb(nb * d);
    std::vector<float> xq(nq * d);
    fill_random(xb.data(), nb, d, 123);
    fill_random(xq.data(), nq, d, 321);

    // Build Flat index
    IndexFlatL2 index(d);
    // std::vector<idx_t> ids(nb);
    // for (idx_t i = 0; i < (idx_t)nb; ++i) ids[i] = i;
    // index.add_with_ids(nb, xb.data(), ids.data());
    index.add(nb, xb.data());


    // Per-query allow-lists
    // q=0: even ids
    // q=1: odd  ids
    // q=2: ids divisible by 3
    auto is_allowed = [&](size_t q, idx_t id) {
        if (q == 0) return (id % 2 == 0);
        if (q == 1) return (id % 2 == 1);
        return (id % 3 == 0);
    };

    std::vector<std::unordered_set<idx_t>> allow_sets(nq);
    for (size_t q = 0; q < nq; ++q) {
        for (idx_t id = 0; id < (idx_t)nb; ++id) {
            if (is_allowed(q, id)) allow_sets[q].insert(id);
        }
    }
    IDSelector2DHashed sel2d_hashed(std::move(allow_sets));

    const size_t bytes_per_bitmap = (nb + 7) / 8;
    std::vector<uint8_t> bitmap_data(nq * bytes_per_bitmap, 0);
    for (size_t q = 0; q < nq; ++q) {
        uint8_t* bm = bitmap_data.data() + q * bytes_per_bitmap;
        for (idx_t id = 0; id < (idx_t)nb; ++id) {
            if (is_allowed(q, id)) bm[id >> 3] |= (1u << (id & 7));
        }
    }
    IDSelector2DBitmap sel2d_bitmap;
    sel2d_bitmap.base = bitmap_data.data();
    sel2d_bitmap.bytes_per_bitmap = bytes_per_bitmap;
    sel2d_bitmap.N = nb;
    sel2d_bitmap.nq = nq;

    SearchParametersSelector2D params2d;

    std::vector<float> D(nq * k);
    std::vector<idx_t> I(nq * k);

    auto validate_results = [&](const char* tag) {
        printf("Results for %s:\n", tag);
        bool all_valid = true;
        for (size_t q = 0; q < nq; ++q) {
            printf("Query %zu: ", q);
            bool ok = true;
            for (size_t j = 0; j < k; ++j) {
                idx_t id = I[q * k + j];
                float dis = D[q * k + j];
                if (id >= 0) {
                    if (!is_allowed(q, id)) {
                        printf("INVALID(id=%" PRId64 ") ", id);
                        ok = false;
                    } else {
                        printf("(%.3f,%" PRId64 ") ", dis, id);
                    }
                } else {
                    printf("(%.3f,%ld) ", dis, (long)id);
                }
            }
            printf("%s\n", ok ? "✓" : "✗");
            all_valid = all_valid && ok;
        }
        return all_valid;
    };

    // Test Hashed 2D
    printf("\n--- Testing IDSelector2DHashed ---\n");
    params2d.sel2d = &sel2d_hashed;
    std::fill(D.begin(), D.end(), 0.f);
    std::fill(I.begin(), I.end(), -1);
    index.search(nq, xq.data(), k, D.data(), I.data(), &params2d);
    bool hashed_ok = validate_results("IDSelector2DHashed");

    // Test Bitmap 2D
    printf("\n--- Testing IDSelector2DBitmap ---\n");
    params2d.sel2d = &sel2d_bitmap;
    std::fill(D.begin(), D.end(), 0.f);
    std::fill(I.begin(), I.end(), -1);
    index.search(nq, xq.data(), k, D.data(), I.data(), &params2d);
    bool bitmap_ok = validate_results("IDSelector2DBitmap");

    printf("\n--- Testing IDSelectorFrom2DForQ ---\n");
    {
        // re-create allow sets for adapter test
        std::vector<std::unordered_set<idx_t>> test_sets(nq);
        for (size_t q = 0; q < nq; ++q) {
            for (idx_t id = 0; id < (idx_t)nb; ++id) {
                if (is_allowed(q, id)) test_sets[q].insert(id);
            }
        }
        IDSelector2DHashed test_sel2d(std::move(test_sets));
        for (size_t q = 0; q < nq; ++q) {
            IDSelectorFrom2DForQ a(&test_sel2d, (idx_t)q);
            printf("Query %zu adapter: ", q);
            int pass = 0;
            for (idx_t id = 0; id < std::min<idx_t>(10, nb); ++id) {
                bool exp = is_allowed(q, id);
                bool act = a.is_member(id);
                if (exp == act) pass++; else {
                    printf("FAIL(id=%" PRId64 " exp=%d act=%d) ", id, (int)exp, (int)act);
                }
            }
            printf("%d/10 tests passed\n", pass);
        }
    }

    printf("\n--- Testing AND(sel1d, sel2d) ---\n");
    struct GlobalAllowLessThan : IDSelector {
        idx_t cutoff;
        explicit GlobalAllowLessThan(idx_t c) : cutoff(c) {}
        bool is_member(idx_t id) const final override { return id >= 0 && id < cutoff; }
    } global_lt((idx_t)nb - 7);

    SearchParametersSelector2D params_and;
    params_and.sel2d = &sel2d_bitmap;    
    params_and.sel   = &global_lt;       

    std::fill(D.begin(), D.end(), 0.f);
    std::fill(I.begin(), I.end(), -1);
    index.search(nq, xq.data(), k, D.data(), I.data(), &params_and);

    bool and_ok = true;
    for (size_t q = 0; q < nq; ++q) {
        for (size_t j = 0; j < k; ++j) {
            idx_t id = I[q * k + j];
            if (id < 0) continue;
            if (!(is_allowed(q, id) && id < (idx_t)nb - 7)) {
                and_ok = false;
            }
        }
    }
    printf("AND(sel1d, sel2d): %s\n", and_ok ? "PASSED" : "FAILED");


    // 9) Final summary
    printf("\n=== Test Results ===\n");
    printf("IDSelector2DHashed: %s\n", hashed_ok ? "PASSED" : "FAILED");
    printf("IDSelector2DBitmap: %s\n", bitmap_ok ? "PASSED" : "FAILED");
    printf("AND(sel1d, sel2d):  %s\n", and_ok    ? "PASSED" : "FAILED");

    bool all_ok = hashed_ok && bitmap_ok && and_ok;
    if (all_ok) {
        printf("All IndexFlat 2D selector tests PASSED! ✓\n");
        return 0;
    } else {
        printf("Some tests FAILED! ✗\n");
        return 1;
    }
}
