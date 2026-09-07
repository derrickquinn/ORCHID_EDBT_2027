"""Console-only progress for offline setup and calibration."""

from contextlib import contextmanager, nullcontext
from time import perf_counter


class Progress:
    def __init__(self):
        self.totals = {}

    @contextmanager
    def task(self, label, *, verbose=False):
        start = perf_counter()
        try:
            yield self
        finally:
            elapsed = perf_counter() - start
            self.totals[label] = self.totals.get(label, 0.0) + elapsed
            if verbose:
                print(f"{label} ({elapsed:.2f}s)", flush=True)

    def progress(self, label, iterable, *, total=None, verbose=False):
        if verbose:
            print(label, flush=True)
        yield from iterable

    def paused(self):
        return nullcontext()

    def print_summary(self, *, file=None):
        for label, elapsed in self.totals.items():
            print(f"{label}: {elapsed:.3f}s", file=file)
