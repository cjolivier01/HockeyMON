from pathlib import Path

from hm_bazel_runtime import launch

if __name__ == "__main__":
    launch("hmlib.cli.hstream_backend", Path(__file__).absolute().parents[1])
