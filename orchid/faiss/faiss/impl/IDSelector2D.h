/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <vector>
#include <unordered_set>

#include <faiss/Index.h>  // for idx_t
#include <faiss/impl/IDSelector.h>  // for IDSelector base class

/** Query-aware ID selectors: the membership decision can depend on
 *  both the query index (q) and the database vector id (id).
 *
 *  Intended to enable per-query filtering in a single batched
 *  index.search(nq, ...) call.
 */

namespace faiss {

/** Base class: query-aware membership test. */
struct IDSelector2D {
    virtual bool is_member(idx_t q, idx_t id) const = 0;
    virtual ~IDSelector2D() {}
};

/** Fast per-query bitmap selector.
 *
 * Layout: nq bitmaps back-to-back in a single blob:
 *   [ bm(0) | bm(1) | ... | bm(nq-1) ]
 *
 * Each bitmap has size ceil(N / 8) bytes, where N is the universe
 * size (max id + 1). Bit i of bm(q) indicates whether id=i is allowed
 * for query q.
 *
 * The selector does not own the memory; the caller must keep it alive
 * for the duration of the search.
 */
struct IDSelector2DBitmap : IDSelector2D {
    const uint8_t* base = nullptr;    ///< pointer to concatenated bitmaps
    size_t bytes_per_bitmap = 0;      ///< ceil(N / 8)
    idx_t N = 0;                      ///< universe size (id in [0, N))
    idx_t nq = 0;                     ///< number of queries

    inline const uint8_t* bm(idx_t q) const {
        return base + q * bytes_per_bitmap;
    }

    bool is_member(idx_t q, idx_t id) const final {
        // (optional) bounds checks can be added in debug builds
        const uint8_t* b = bm(q);
        return (b[id >> 3] >> (id & 7)) & 1;
    }
    ~IDSelector2DBitmap() override {}
};

/** Per-query hashed allow lists.
 *
 * Use when each query has a sparse set of allowed ids.
 * allow[q] is a hash set of ids permitted for query q.
 */
struct IDSelector2DHashed : IDSelector2D {
    std::vector<std::unordered_set<idx_t>> allow;

    explicit IDSelector2DHashed(std::vector<std::unordered_set<idx_t>> allow)
        : allow(std::move(allow)) {}

    bool is_member(idx_t q, idx_t id) const final {
        if (q < 0 || q >= (idx_t)allow.size()) {
            return false;  // Invalid query index
        }
        const auto& S = allow[q];
        return S.find(id) != S.end();
    }
    ~IDSelector2DHashed() override {}
};

/** Convenience wrapper: adapt a 2D selector to the 1D IDSelector
 *  interface for a fixed query index q. Useful internally if a code
 *  path expects an IDSelector but you need per-query filtering.
 */
struct IDSelectorFrom2DForQ : IDSelector {
    const IDSelector2D* sel2d;
    idx_t q;

    IDSelectorFrom2DForQ(const IDSelector2D* sel2d, idx_t q)
        : sel2d(sel2d), q(q) {
        // printf("DEBUG: Creating adapter for query %ld\n", q);
    }

    bool is_member(idx_t id) const final {
        bool result = !sel2d || sel2d->is_member(q, id);
        // printf("DEBUG: Adapter q=%ld checking id=%ld -> %s\n", 
        //        q, id, result ? "ALLOW" : "REJECT");
        return result;
    }
    ~IDSelectorFrom2DForQ() override {}
};

} // namespace faiss