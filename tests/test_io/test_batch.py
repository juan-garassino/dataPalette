from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from datapalette.io import load_image, process_directory, save_image


class TestLoadImage:
    def test_loads_valid_image(self, tmp_path):
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        path = str(tmp_path / "test.png")
        cv2.imwrite(path, img)
        loaded = load_image(path)
        assert loaded.shape == (32, 32, 3)
        assert loaded.dtype == np.uint8

    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/path/image.png")


class TestSaveImage:
    def test_saves_and_creates_dirs(self, tmp_path):
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        path = str(tmp_path / "subdir" / "out.png")
        save_image(path, img)
        assert os.path.isfile(path)
        loaded = cv2.imread(path)
        assert loaded is not None
        assert loaded.shape == (32, 32, 3)


class TestProcessDirectory:
    def test_processes_all_images(self, tmp_path):
        in_dir = str(tmp_path / "input")
        out_dir = str(tmp_path / "output")
        os.makedirs(in_dir)

        for i in range(3):
            img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(in_dir, f"img_{i}.png"), img)

        def flip(img):
            return cv2.flip(img, 1)

        saved = process_directory(in_dir, out_dir, flip)
        assert len(saved) == 3
        for path in saved:
            assert os.path.isfile(path)

    def test_empty_directory(self, tmp_path):
        in_dir = str(tmp_path / "empty_in")
        out_dir = str(tmp_path / "empty_out")
        os.makedirs(in_dir)
        saved = process_directory(in_dir, out_dir, lambda img: img)
        assert len(saved) == 0
