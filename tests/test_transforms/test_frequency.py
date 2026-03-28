from __future__ import annotations

import numpy as np

from datapalette.transforms import FourierTransform


class TestFourierTransform:
    def test_output_shape_dtype_3channel(self, sample_image):
        t = FourierTransform()
        result = t.transform(sample_image)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_output_shape_dtype_single_channel(self):
        gray = np.random.randint(0, 255, (64, 64, 1), dtype=np.uint8)
        t = FourierTransform()
        result = t.transform(gray)
        assert result.shape == (64, 64, 1)
        assert result.dtype == np.uint8
