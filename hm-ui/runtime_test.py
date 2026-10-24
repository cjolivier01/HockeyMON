"""Exercise built launchers without dependencies in Bazel's build-tool Python."""

import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.binaries = Path(os.environ["TEST_SRCDIR"]) / os.environ["TEST_WORKSPACE"] / "hm-ui"
        self.environment = dict(os.environ, PYTHON_BIN_PATH=sys.executable, CONDA_PREFIX="/invalid")
        self.environment.pop("HM_CLI_BIN", None)

    def run_binary(self, name, *arguments):
        return subprocess.run(
            [str(self.binaries / name), *arguments],
            check=False,
            env=self.environment,
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=20,
        )

    def test_explicit_interpreter_and_lightweight_entrypoints(self):
        ui = self.run_binary("hm-ui", "--help")
        self.assertEqual(ui.returncode, 0, ui.stderr)
        self.assertIn("shared HStream desktop", ui.stdout)
        cli = self.run_binary("hm-cli", "--ui-backend-info")
        self.assertEqual(cli.returncode, 0, cli.stderr)
        self.assertEqual(json.loads(cli.stdout)["backend"], "hockeymon")

    def test_built_desktop_selects_its_built_backend(self):
        desktop = self.root / "desktop"
        desktop.write_text('#!/bin/sh\nprintf "%s\\n" "$HM_CLI_BIN"\n')
        desktop.chmod(0o755)
        self.environment["HSTREAM_UI_BIN"] = str(desktop)
        result = self.run_binary("hm-ui", "--game-id", "example")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            Path(result.stdout.strip()).resolve(), (self.binaries / "hm-cli").resolve()
        )

    def test_active_conda_and_path_interpreters(self):
        prefix = self.root / "environment"
        (prefix / "bin").mkdir(parents=True)
        marker = self.root / "selected"
        interpreter = prefix / "bin" / "python"
        interpreter.write_text(
            "#!/bin/sh\n"
            f"touch {shlex.quote(str(marker))}\n"
            f'exec {shlex.quote(sys.executable)} "$@"\n'
        )
        interpreter.chmod(0o755)
        self.environment.pop("PYTHON_BIN_PATH")
        self.environment["CONDA_PREFIX"] = str(prefix)
        self.assertEqual(self.run_binary("hm-ui", "--help").returncode, 0)
        self.assertTrue(marker.is_file())
        marker.unlink()
        (prefix / "bin" / "python3").symlink_to(interpreter)
        self.environment.pop("CONDA_PREFIX")
        self.environment["PATH"] = str(prefix / "bin") + os.pathsep + os.environ["PATH"]
        self.assertEqual(self.run_binary("hm-ui", "--help").returncode, 0)
        self.assertTrue(marker.is_file())

    def test_invalid_explicit_interpreter_does_not_fall_back(self):
        self.environment["PYTHON_BIN_PATH"] = str(self.root / "missing-python")
        result = self.run_binary("hm-ui", "--help")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("selected HockeyMON Python is not executable", result.stderr)


if __name__ == "__main__":
    unittest.main()
