from __future__ import annotations

import math
import os

from ariadnepy.exceptions import AriadneError


def get_batches(
    x_len: int | None,
    batch_size: int,
    workers: int | None,
    factor: int = 3,
) -> list[tuple[int, int]]:
    """Split a sequence of length ``x_len`` into (start, end) index ranges.

    Mirrors R ariadne's internal ``.get_batches``: the number of batches is
    capped at ``factor * workers`` so that batching never spawns more jobs
    than the worker pool can use, then raises if the resulting per-batch size
    still exceeds ``batch_size``.

    Parameters
    ----------
    x_len:
        Length of the sequence to batch, or None for a single empty batch.
    batch_size:
        Maximum number of items per batch.
    workers:
        Number of parallel workers. Auto-detected from CPU count when None.
    factor:
        Number of jobs per worker.

    Returns
    -------
    list of (start, end)
        0-based, end-exclusive index ranges suitable for slicing.
    """
    if x_len is None:
        return [(0, 0)]

    if workers is None:
        workers = os.cpu_count() or 1

    batch_num = max(min(math.ceil(x_len / batch_size), factor * workers), 1)
    adapted_size = math.ceil(x_len / batch_num)

    if adapted_size > batch_size:
        raise AriadneError(
            f"Query limit was reached ({adapted_size} > {batch_size}). "
            "Increase 'factor', 'batch_size' or 'workers' and try again."
        )

    return [
        (i * adapted_size, min((i + 1) * adapted_size, x_len))
        for i in range(batch_num)
    ]
