from __future__ import annotations

import subprocess
import sys

from datapalette.cli import main


class TestCLI:
    def test_main_is_importable(self):
        assert callable(main)

    def test_help_flag(self):
        result = subprocess.run(
            [sys.executable, "-m", "datapalette.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "DataPalette" in result.stdout
        assert "--pipeline" in result.stdout
