from __future__ import annotations

import numpy as np

from datapalette.transforms import EdgeChannels, GradientChannels


class TestGradientChannels:
    def test_output_2_channels(self, sample_image):
        t = GradientChannels()
        result = t.transform(sample_image)
        assert result.shape == (100, 100, 2)
        assert result.dtype == np.uint8


class TestEdgeChannels:
    def test_output_2_channels(self, sample_image):
        t = EdgeChannels()
        result = t.transform(sample_image)
        assert result.shape == (100, 100, 2)
        assert result.dtype == np.uint8

    def test_custom_thresholds(self, sample_image):
        t = EdgeChannels(low_threshold=50, high_threshold=150)
        result = t.transform(sample_image)
        assert result.shape == (100, 100, 2)
