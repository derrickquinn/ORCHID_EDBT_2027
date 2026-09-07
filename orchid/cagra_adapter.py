"""Stable Python surface for the native cuVS filtered-CAGRA adapter."""

from _cagra_native import Index, Metric, artifact_repository, artifact_revision


# The adapter is derived from this immutable upstream artifact revision. Local
# adapter changes are already represented by the benchmark's source git state.
baseline_revision = artifact_revision
baseline_dirty = False


__all__ = [
    "Index",
    "Metric",
    "artifact_repository",
    "artifact_revision",
    "baseline_dirty",
    "baseline_revision",
]
