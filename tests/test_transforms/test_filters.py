from __future__ import annotations

import numpy as np

from datapalette.transforms import CustomKernel, Emboss, Sharpen


class TestEmboss:
    def test_output_shape(self, sample_image):
        t = Emboss()
        result = t.transform(sample_image)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8


class TestSharpen:
    def test_output_shape(self, sample_image):
        t = Sharpen()
        result = t.transform(sample_image)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8


class TestCustomKernel:
    def test_default_kernel_output_shape(self, sample_image):
        t = CustomKernel()
        result = t.transform(sample_image)
        assert result.shape == sample_image.shape

    def test_custom_kernel_output_shape(self, sample_image):
        kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
        t = CustomKernel(kernel=kernel)
        result = t.transform(sample_image)
        assert result.shape == sample_image.shape
