from pathlib import Path

from hm_bazel_runtime import launch

if __name__ == "__main__":
    launch("hmlib.ui.desktop", Path(__file__).absolute().parents[1])
