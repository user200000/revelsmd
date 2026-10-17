"""Measure the memory an object keeps, with tracemalloc."""

import gc
import tracemalloc


def retained_bytes(build):
    """Bytes still allocated after ``build()`` returns, while its result is kept.

    Memory freed before ``build`` returns is not counted. NumPy array data
    is traced along with Python objects.
    """
    gc.collect()
    started = not tracemalloc.is_tracing()
    if started:
        tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        kept = build()
        gc.collect()
        after = tracemalloc.get_traced_memory()[0]
    finally:
        if started:
            tracemalloc.stop()
    del kept
    return after - before


def bytes_per_extra_item(build, n):
    """Retained bytes per item of ``build(2 * n)`` beyond ``build(n)``.

    ``build(n)`` runs once first so that one-off caches and imports are not
    counted.
    """
    build(n)
    small = retained_bytes(lambda: build(n))
    large = retained_bytes(lambda: build(2 * n))
    return (large - small) / n
