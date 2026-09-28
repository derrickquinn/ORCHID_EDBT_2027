# SIFT-12 reference data

`sift12-reference.npz` is a 794,187-byte, pickle-free NumPy archive that fixes
otherwise unrecoverable choices in the local SIFT-12 workload. It contains:

- `rotation`: a float64 128 × 128 matrix applied as `source @ rotation` before
  conversion to float32. The original rotation used an unseeded QR decomposition
  and its matrix was not saved. This matrix was recovered by float64 least
  squares from the first 8,192 ANN-Benchmarks training vectors and their local
  rotated counterparts. Across all 1,010,000 train/test vectors, the largest
  absolute difference from the local float32 stores was `1.52587890625e-5`.
  Vectors retain their original magnitudes; the evaluation harness normalizes.
- `neighbors`: the local `sift1m_c12.zarr/neighbors`, int64 10,000 × 10,
  for the seed-42 equality workload and cosine similarity. These IDs and their
  original ordering are preserved, including numerical ties.
- `neighbors_unfiltered`: the local `sift1m.zarr/neighbors_unfiltered`,
  int64 10,000 × 10, computed with squared L2 on unnormalized vectors. This
  auxiliary array is retained for completeness; the SIFT-12 benchmark uses
  the filtered `neighbors` array.

The reference was captured on 2026-09-28. Its SHA-256 is
`7f07540e114a0839147bc106282e13b5eca4bd9f0250eeb6ee06abd3cbac6fa2`.
The script validates this checksum before preparing data. Keeping these small
neighbor tables avoids expensive exhaustive recomputation and changes to
float-dependent tie ordering. ANN-Benchmarks' supplied Euclidean neighbor table
is not the ground truth for the filtered cosine workload.

The source is [ANN-Benchmarks SIFT](https://ann-benchmarks.com/sift-128-euclidean.hdf5),
525,128,288 bytes, SHA-256
`dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984`.
See the [original SIFT dataset](http://corpus-texmex.irisa.fr/) for provenance.

For auditing the recovered rotation, with `X` the first 8,192 source training
rows and `Y` the corresponding original local rotated rows:

```python
rotation = np.linalg.lstsq(X.astype(np.float64), Y.astype(np.float64), rcond=None)[0]
recovered = (source.astype(np.float64) @ rotation).astype(np.float32)
```

Canonical local arrays, SHA-256 of contiguous row-major bytes (little endian):

| Array | SHA-256 |
| --- | --- |
| `sift1m.zarr/train` | `fca1f13669d2c7c499c4da75a8c3b25ff231b85fc163942732d790cc483b2264` |
| `sift1m.zarr/test` | `c4a75bef76cd36fc6e831a2f1ba2d76921bbae5b72f6de6def0a04bea7e7941a` |

These are provenance hashes of the original arrays, not expected hashes of the
recovered vectors. Small float32 differences are intentional and accepted;
the mask values and neighbor IDs must remain exactly equal.
