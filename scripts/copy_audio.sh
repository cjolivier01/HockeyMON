#!/bin/bash
set -e
INPUT_VIDEO_WITH_AUDIO="$1"
INPUT_VIDEO_NO_AUDIO="$2"
OUTPUT_VIDEO_WITH_AUDIO="$3"
shift 3
ffmpeg \
  -i "${INPUT_VIDEO_WITH_AUDIO}" \
  -i "${INPUT_VIDEO_NO_AUDIO}" \
  -c:v copy -c:a copy \
  -map 1:v:0 \
  -map 0:a:0 \
  -shortest \
  "$@" \
  "${OUTPUT_VIDEO_WITH_AUDIO}"
