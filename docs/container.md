# HockeyMON CUDA Container (User Guide)

This repo provides a CUDA-enabled Docker image that includes HockeyMON (native extension + Python wheels) and its CLI tools.

## Prerequisites (host)

- Docker installed and working.
- For GPU runs: NVIDIA driver + NVIDIA Container Toolkit (`docker run --gpus all ...` must work).
- Videos live under `~/Videos/<game-id>/...` by default.

## Build

Build the image:

```bash
./DockerBuild.sh
```

Specify a tag explicitly:

```bash
./DockerBuild.sh --tag hm
```

## Run

Run an interactive shell:

```bash
./DockerRun.sh bash
```

Run tracking (example):

```bash
./DockerRun.sh hmtrack --game-id stockton-r3 -t=60
```

Run tracking with a host working directory mounted as the container cwd:

```bash
./DockerRun.sh --workdir /path/to/hm-work hmtrack --game-id stockton-r3
```

With `--workdir`, relative outputs (such as `./output_workdirs/...`) are written under the mounted host directory.

Run stitching (example):

```bash
./DockerRun.sh hmstitch --game-id stockton-r3 -o stitched.mp4
```

## Videos mount

By default, the wrapper mounts your host `~/Videos` into the container so tools can use `$HOME/Videos/<game-id>`.

Override the mount:

```bash
./DockerRun.sh --videos-mount /path/to/Videos hmtrack --game-id stockton-r3 -t=60
```

Disable mounting:

```bash
./DockerRun.sh --no-videos-mount bash
```

## Workdir mount

Use `--workdir <dir>` to mount a host directory at `/workspace/workdir` and make it the container working directory. This is useful when you want `output_workdirs` on a host path you choose.

Example:

```bash
./DockerRun.sh --workdir /mnt/data/hm-runs hmtrack --game-id stockton-r3
```

## Stitching + NVENC width limit (8192)

When `hmstitch` writes a video using NVENC, panoramas wider than 8192 pixels can fail on many GPUs.

- `./DockerRun.sh hmstitch ... -o out.mp4` auto-adds:
  - `--config-override aspen.stitching.max_output_width=8192`
- Disable the auto-clamp with:
  - `./DockerRun.sh hmstitch --no-hwenc-clamp ...`

`hmtrack` does not need this clamp for typical workflows because it does not encode the full stitched panorama as the output video.

## CLI docs

The main CLI commands are `hmtrack` and `hmstitch`. The full list and usage notes live here:

- `hmlib/cli/README.md`

List installed CLIs inside the container:

```bash
./DockerRun.sh list-cli
```

## Open docs

Print doc paths, or open them in your browser (via `xdg-open`):

```bash
./DockerRun.sh docs
./DockerRun.sh docs --open
```

If you are in a headless terminal (no GUI `DISPLAY`), serve the docs locally and open the URL in your browser:

```bash
./DockerRun.sh docs --serve
```
