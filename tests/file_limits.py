"""Helpers for tests that need a low open-file limit."""

import os
from contextlib import contextmanager

import pytest


def open_descriptors():
    """Number of file descriptors this process has open."""
    if not os.path.isdir("/dev/fd"):
        pytest.skip("needs /dev/fd to count open files")
    return len(os.listdir("/dev/fd"))


@contextmanager
def open_file_limit(limit):
    """Lower the soft open-file limit for the duration of the block."""
    resource = pytest.importorskip("resource")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
