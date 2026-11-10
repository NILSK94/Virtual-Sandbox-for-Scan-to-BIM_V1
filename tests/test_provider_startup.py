"""Regression test for native argument quoting in the Windows launcher."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scanbim.settings import load_credentials, save_settings


@unittest.skipUnless(os.name == "nt", "Windows PowerShell launcher")
class ProviderStartupTests(unittest.TestCase):
    def test_actual_startup_switch_preserves_credentials_and_consumes_marker(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root) as directory:
            folder = Path(directory)
            with patch.dict(os.environ, {"SCANBIM_SETTINGS_FILE": str(folder / "settings.toml")}):
                save_settings("test-private-key", "gpt-6.1-sol", "openai")
                save_settings("", "qwen3.5:4b", "ollama")
                marker = folder / "provider-next-start.txt"
                marker.write_text("openai\n", encoding="utf-8")
                source = (root / "Start.ps1").read_text(encoding="utf-8")
                block = source[source.index("$pendingProvider ="):source.index("$studioProvider =")]
                script = folder / "switch.ps1"
                script.write_text("$ErrorActionPreference = 'Stop'\n$venvPython = '" +
                                  sys.executable.replace("'", "''") + "'\n" + block, encoding="utf-8-sig")
                run = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                                      "-File", str(script)], cwd=root, capture_output=True, text=True, timeout=30)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertFalse(marker.exists())
                credentials = load_credentials()
                self.assertEqual(credentials.provider, "openai")
                self.assertEqual(credentials.model, "gpt-6.1-sol")
                self.assertEqual(credentials.api_key, "test-private-key")
                self.assertNotIn("test-private-key", run.stdout + run.stderr)


if __name__ == "__main__":
    unittest.main()
