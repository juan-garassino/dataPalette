from __future__ import annotations

import os

import cv2
import numpy as np

from datapalette.pipelines import SegmentationPipeline


class TestSegmentationPipeline:
    def test_load_and_transform_custom(self, tmp_path):
        images_dir = str(tmp_path / "images")
        masks_dir = str(tmp_path / "masks")
        os.makedirs(images_dir)
        os.makedirs(masks_dir)

        for i in range(3):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            mask = np.random.randint(0, 2, (64, 64, 3), dtype=np.uint8) * 255
            cv2.imwrite(os.path.join(images_dir, f"img_{i:03d}.png"), img)
            cv2.imwrite(os.path.join(masks_dir, f"img_{i:03d}.png"), mask)

        pipe = SegmentationPipeline(dataset=None, mode="binary", size=(32, 32))
        X, y = pipe.load_and_transform(images_dir=images_dir, masks_dir=masks_dir)
        assert len(X) == 3
        assert len(y) == 3
        for img in X:
            assert img.shape == (32, 32, 3)
        for mask in y:
            assert mask.shape == (32, 32, 3)

    def test_raises_without_masks_dir(self, tmp_image_dir):
        pipe = SegmentationPipeline(dataset=None)
        try:
            pipe.load_and_transform(images_dir=tmp_image_dir, masks_dir=None)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "masks_dir" in str(e)
