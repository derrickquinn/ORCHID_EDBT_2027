import itertools
from functools import cache
import numpy as np
import numba as nb


@nb.njit(parallel=True, cache=True, fastmath=True)
def _fast_sort(a, work):
    b = np.empty(a.shape, dtype=np.int32)

    for thread_work in nb.prange(work.shape[0]):
        for arr in work[thread_work]:
            i = arr[0]
            j = arr[1]
            b[i, j, :] = np.argsort(a[i, j])[::-1]
    return b


@nb.njit(parallel=True, cache=True, fastmath=True)
def _fast_cumsum(a: np.ndarray, work: np.ndarray):
    b = np.empty(a.shape, dtype=a.dtype)

    for block_id in nb.prange(work.shape[0]):
        for arr in work[block_id]:
            i = arr[0]
            j = arr[1]
            b[i, j, :] = np.cumsum(a[i, j, :])
    return b


@nb.njit(parallel=True, cache=True, fastmath=True)
def _fast_reindex2d(source, rankings, work):
    c = np.empty(rankings.shape, dtype=source.dtype)

    for block_id in nb.prange(work.shape[0]):
        for ij in work[block_id]:
            i = ij[0]
            j = ij[1]
            ranking = rankings[i, j, :]

            c[i, j] = source[j, ranking]

    return c


@nb.njit(parallel=True, cache=True, fastmath=True)
def _fast_reindex1d(source, rankings, work):
    c = np.empty(rankings.shape, dtype=source.dtype)

    for block_id in nb.prange(work.shape[0]):
        for ij in work[block_id]:
            i = ij[0]
            j = ij[1]
            ranking = rankings[i, j, :]

            c[i, j] = source[ranking]

    return c


# all_ids: [nlist, nci: variable]
# maskT: [N, Q]
# sum(nci) = N
# returns: [nlist, Q, nci]
@nb.njit(parallel=True, cache=True, fastmath=True)
def _fast_cluster_legals(maskT, all_ids, work):
    c = np.empty(
        (
            len(all_ids),
            maskT.shape[1],
        ),
        dtype=np.float32,
    )

    for block_id in nb.prange(work.shape[0]):  # each block has list ids.
        for ij in work[block_id]:  # each list
            i = ij[0]  # this will actually just be an i. should refactor?
            ids_maskT = maskT[all_ids[i], :]
            c[i, :] = np.array(
                [ids_maskT[:, qi].mean() for qi in range(maskT.shape[1])],
                dtype=np.float32,
            )

    return c


@nb.njit(parallel=True, cache=True, fastmath=True)
def _fast_cluster_ids(clusters, nlist, work):
    c = [np.empty(0, dtype=clusters.dtype) for _ in np.arange(nlist)]

    for block_id in nb.prange(work.shape[0]):  # each block has list ids.
        for ij in work[block_id]:
            i = ij[0]
            c[i] = np.flatnonzero(clusters == i)
    return c


@cache
def flat_iter(shape):
    idxs = [np.arange(d, dtype=np.int32) for d in shape[:-1]]
    iterator = itertools.product(*idxs)
    work = np.array(list(iterator), dtype=np.int32)
    return work


@cache
def make_work(shape, threads=32):
    work = flat_iter(shape)
    return np.ascontiguousarray(work.reshape(threads, -1, work.shape[-1]))


def fastsort(a, threads=32):
    work = make_work(a.shape, threads=threads)
    return _fast_sort(a, work)


def fastcumsum(a, threads=32):
    work = make_work(a.shape, threads=threads)
    return _fast_cumsum(a, work)


def fastreindex2d(source, rankings, threads=32):
    work = make_work(rankings.shape, threads=threads)
    return _fast_reindex2d(source, rankings, work)


def fastreindex1d(source, rankings, threads=32):
    work = make_work(rankings.shape, threads=threads)
    return _fast_reindex1d(source, rankings, work)


def fast_cluster_legals(maskT, all_ids, threads=32):
    shape = (len(all_ids), 1)
    work = make_work(shape, threads=threads)
    return _fast_cluster_legals(maskT, all_ids, work)


def fast_cluster_ids(clusters, nlist, threads=32):
    shape = (nlist, 1)
    work = make_work(shape, threads=threads)
    return _fast_cluster_ids(clusters, nlist, work)
