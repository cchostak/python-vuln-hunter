"""Recursively scan directories for Python files."""
import os
from typing import Iterator


def iter_python_files(root: str) -> Iterator[str]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)
