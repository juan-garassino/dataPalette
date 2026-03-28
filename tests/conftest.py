from __future__ import annotations

import os

import cv2
import numpy as np
import pytest


@pytest.fixture
def sample_image() -> np.ndarray:
    """100x100 BGR uint8 image with random pixel values."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def small_image() -> np.ndarray:
    """10x10 BGR uint8 image with random pixel values."""
    return np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)


@pytest.fixture
def tmp_image_dir(tmp_path) -> str:
    """Create a temporary directory containing 3 sample PNG images.

    Returns the path to the directory as a string.
    """
    for i in range(3):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        path = os.path.join(str(tmp_path), f"sample_{i:03d}.png")
        cv2.imwrite(path, img)
    return str(tmp_path)
