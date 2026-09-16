#!/bin/bash

#EXPERIMENT_FILE="$(pwd)/config/models/hm/hm_bytetrack.py"
# EXPERIMENT_FILE="$(pwd)/config/models/hm/hm_end_to_end.py"

#START_FRAME=0

# BATCH_SIZE=1

#WRAPPER_CMD="nsys profile --show-outputs=true --wait=primary --trace=cuda,nvtx,cublas,cudnn,openacc --python-sampling=true --python-backtrace=cuda"
#WRAPPER_CMD="nsys profile"
#WRAPPER_CMD="echo"

#STITCHING_ARGS="--save-stitched"
# Legacy hmtrack flags for saving detections/tracks were removed, so only use the
# still-supported camera CSV option here to avoid CLI errors.
SAVE_DATA_ARGS="--save-camera-data"

echo "Experiment name: ${EXP_NAME}"

if [ ! -z "${VIDEO}" ]; then
  VIDEO="--input-video=${VIDEO}"
fi

REPO_PYTHONPATH="$(pwd)"
if [ -d "$(pwd)/src" ]; then
  REPO_PYTHONPATH="${REPO_PYTHONPATH}:$(pwd)/src"
fi

# MMCV is installed as a compiled package. Adding openmm/mmcv here shadows that
# package with the source tree, where mmcv._ext is not installed. The remaining
# vendored OpenMMLab packages are pure Python and are expected to be imported
# from source during local runs.
OPENMM_PYTHONPATH="$(pwd)/openmm/mmengine:$(pwd)/openmm/mmeval:$(pwd)/openmm/mmdetection:$(pwd)/openmm/mmpose:$(pwd)/openmm/mmocr:$(pwd)/openmm/mmsegmentation:$(pwd)/openmm/mmpretrain:$(pwd)/openmm/mmyolo:$(pwd)/openmm/mmaction2"
PYTHON_SITE_PACKAGES="$("${CONDA_PREFIX}/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
NATIVE_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/hugin:${PYTHON_SITE_PACKAGES}/torch/lib:${PYTHON_SITE_PACKAGES}/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH}"
set -x
OMP_NUM_THREADS=16 \
  LD_LIBRARY_PATH="${NATIVE_LIBRARY_PATH}" \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="${OPENMM_PYTHONPATH}:${REPO_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}" \
  ${WRAPPER_CMD} python -m hmlib.cli.hmtrack \
  ${SAVE_DATA_ARGS} \
  ${HYPER_PARAMS} ${STITCHING_PARAMS} ${TEST_SIZE_ARG} \
  ${VIDEO} $@
status=$?
set +x
if [ "${status}" -ne 0 ]; then
  if [ "${status}" -gt 128 ]; then
    signal=$((status - 128))
    echo "hmtrack exited with status ${status} (signal ${signal})" >&2
  else
    echo "hmtrack exited with status ${status}" >&2
  fi
fi
exit "${status}"
