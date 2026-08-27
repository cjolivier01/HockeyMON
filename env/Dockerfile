FROM nvidia/cuda:13.0.2-devel-ubuntu22.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

ARG USERNAME=hm
ARG UID=1000
ARG GID=1000

# Keep bash in RUN steps so we can source conda, use braces, etc.
SHELL ["/bin/bash", "-lc"]

####################################################################
#
# Base system deps (build + runtime)
#
####################################################################
RUN apt-get update && apt-get install -y --no-install-recommends \
  ca-certificates \
  ccache \
  curl \
  git \
  git-lfs \
  openssh-client \
  sudo \
  wget \
  rsync \
  unzip \
  zip \
  xauth \
  xvfb \
  build-essential \
  pkg-config \
  yasm \
  nasm \
  autoconf \
  automake \
  cmake \
  ninja-build \
  libtool \
  vim \
  graphviz \
  graphviz-dev \
  gettext \
  libyaml-cpp-dev \
  libprotobuf-dev \
  protobuf-compiler \
  libgl1-mesa-dev \
  libglu1-mesa-dev \
  libgl-dev \
  libglew-dev \
  libjpeg-turbo8-dev \
  libtiff-dev \
  libpng-dev \
  libopenexr-dev \
  libexiv2-dev \
  libfftw3-dev \
  libilmbase-dev \
  liblcms2-dev \
  libsqlite3-dev \
  liblapack-dev \
  libopenblas-dev \
  libomp-dev \
  libboost-all-dev \
  libpano13-dev \
  libglfw3-dev \
  libgsl-dev \
  help2man \
  texlive-latex-base \
  libwxgtk3.0-gtk3-dev \
  libx264-dev \
  libx265-dev \
  libvpx-dev \
  libfdk-aac-dev \
  libmp3lame-dev \
  libopus-dev \
  libass-dev \
  libfreetype6-dev \
  libsdl2-dev \
  libtheora-dev \
  libva-dev \
  libgnutls28-dev \
  libvdpau-dev \
  libvorbis-dev \
  libxcb1-dev \
  libxcb-shm0-dev \
  libxcb-xfixes0-dev \
  libopencv-dev \
  libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev \
  libgstrtspserver-1.0-dev \
  libsoup2.4-dev \
  libjson-glib-dev \
  texinfo \
  zlib1g-dev \
  libavformat-dev \
  libavcodec-dev \
  libavdevice-dev \
  libavutil-dev \
  libswscale-dev \
  libswresample-dev \
  && rm -rf /var/lib/apt/lists/*

####################################################################
#
# Build OpenCV (C++11 ABI) for Torch extensions
#
####################################################################
# PyTorch CUDA wheels are built with `_GLIBCXX_USE_CXX11_ABI=1`, and our native
# extensions link against OpenCV. Build an ABI-compatible OpenCV into /usr/local.
WORKDIR /tmp
RUN git clone https://github.com/opencv/opencv.git \
  && cd opencv \
  && git checkout 4.6.0 \
  && mkdir -p build \
	  && cd build \
	  && cmake -DCMAKE_BUILD_TYPE=Release \
	           -DCMAKE_INSTALL_PREFIX=/usr/local \
	           -DBUILD_SHARED_LIBS=ON \
           -DBUILD_LIST=core,calib3d,features2d,highgui,imgcodecs,imgproc,video,videoio \
           -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
           -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF \
           -DBUILD_JAVA=OFF \
           -DWITH_CUDA=OFF \
           -DWITH_QT=OFF \
           -DWITH_OPENGL=OFF \
           .. \
  && make -j"$(nproc)" \
  && make -j"$(nproc)" install \
  && ldconfig \
  # Ensure linkers/runtime resolve to our ABI-compatible OpenCV in /usr/local
  # (the distro `libopencv-dev` also provides unversioned `libopencv_*.so` in /lib).
  && for lib in core calib3d features2d highgui imgcodecs imgproc video videoio; do \
       ln -sf "/usr/local/lib/libopencv_${lib}.so" "/lib/x86_64-linux-gnu/libopencv_${lib}.so"; \
       ln -sf "/usr/local/lib/libopencv_${lib}.so" "/usr/lib/x86_64-linux-gnu/libopencv_${lib}.so"; \
     done \
  && ldconfig \
  && cd /tmp \
  && rm -rf opencv

####################################################################
#
# User setup
#
####################################################################
RUN set -eux; \
  if getent group "${GID}" >/dev/null 2>&1; then \
    group_name="$(getent group "${GID}" | cut -d: -f1)"; \
  else \
    group_name="${USERNAME}"; \
    groupadd --gid "${GID}" "${group_name}"; \
  fi; \
  if id -u "${USERNAME}" >/dev/null 2>&1; then \
    true; \
  else \
    useradd -m -s /bin/bash --uid "${UID}" --gid "${group_name}" -G sudo "${USERNAME}"; \
  fi; \
  echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >/etc/sudoers.d/"${USERNAME}"; \
  chmod 0440 /etc/sudoers.d/"${USERNAME}"

####################################################################
#
# Conda (needed for Bazel conda_repository + Python runtime)
#
####################################################################
ENV CONDA_DIR=/opt/conda
RUN wget -O /tmp/miniforge.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
  && bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" \
  && rm -f /tmp/miniforge.sh \
  && "${CONDA_DIR}/bin/conda" config --system --set auto_update_conda false \
  && "${CONDA_DIR}/bin/conda" config --system --set show_channel_urls true \
  && "${CONDA_DIR}/bin/conda" create -y -n hm python=3.10 pip \
  && "${CONDA_DIR}/bin/conda" clean -a -y

ENV CONDA_PREFIX=${CONDA_DIR}/envs/hm
ENV PATH=${CONDA_PREFIX}/bin:${CONDA_DIR}/bin:$PATH
# Avoid putting the conda environment first in `LD_LIBRARY_PATH`: it can break
# system tools (e.g. git) by forcing them to load conda's OpenSSL.
# Avoid `/usr/local/cuda/compat` here: it can override the host-mounted `libcuda.so`
# and trigger "unsupported display driver / cuda driver combination" (Error 803).
# Keep Torch's shared libraries discoverable for our native extensions.
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/usr/local/nvidia/lib:${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/python3.10/site-packages/torch/lib

RUN pip install --no-cache-dir -U pip setuptools wheel

####################################################################
#
# Bazelisk (used by ./run.sh and Bazel builds)
#
####################################################################
RUN curl -fsSL -o /usr/local/bin/bazelisk \
    https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-amd64 \
  && chmod +x /usr/local/bin/bazelisk \
  && ln -sf /usr/local/bin/bazelisk /usr/local/bin/bazel

####################################################################
#
# Build FFmpeg with CUDA/NVENC/NVDEC support
#
####################################################################
WORKDIR /tmp
RUN git clone https://github.com/FFmpeg/nv-codec-headers.git \
  && cd nv-codec-headers \
  && git checkout n11.1.5.3 \
  && make -j"$(nproc)" install \
  && cd /tmp \
  && rm -rf nv-codec-headers

RUN git clone https://github.com/FFmpeg/FFmpeg.git ffmpeg \
  && cd ffmpeg \
  && git checkout n5.1.4 \
  && NVCCFLAGS="-gencode arch=compute_75,code=sm_75 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120" \
     ./configure \
       --prefix="/usr/local" \
       --extra-cflags='-I/usr/local/cuda/include' \
       --extra-ldflags='-L/usr/local/cuda/lib64' \
       --disable-doc \
       --enable-decoder=aac \
       --enable-decoder=h264 \
       --enable-decoder=h264_cuvid \
       --enable-decoder=hevc_cuvid \
       --enable-decoder=av1_cuvid \
       --enable-decoder=rawvideo \
       --enable-indev=lavfi \
       --enable-encoder=libx264 \
       --enable-encoder=h264_nvenc \
       --enable-encoder=hevc_nvenc \
       --enable-encoder=av1_nvenc \
       --enable-demuxer=mov \
       --enable-muxer=mp4 \
       --enable-filter=scale \
       --enable-filter=testsrc2 \
       --enable-protocol=file \
       --enable-protocol=https \
       --enable-gnutls \
       --enable-shared \
       --enable-gpl \
       --enable-nonfree \
       --enable-cuda-nvcc \
       --enable-libx264 \
       --enable-libnpp \
       --enable-nvenc \
       --enable-cuvid \
       --enable-nvdec \
       --enable-pic \
       --enable-rpath \
       --extra-cflags=-I/usr/local/cuda/include \
       --extra-ldflags=-L/usr/local/cuda/lib64 \
  && make -j"$(nproc)" \
  && make -j"$(nproc)" install \
  && cd /tmp \
  && rm -rf ffmpeg

####################################################################
#
# Build VIGRA for Enblend/Hugin
#
####################################################################
WORKDIR /tmp
RUN git clone https://github.com/cjolivier01/vigra \
  && cd vigra \
  && git checkout colivier/hm \
  && mkdir -p build \
  && cd build \
  && cmake -DCMAKE_BUILD_TYPE=Release -DWITH_OPENEXR=ON -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_LIBDIR=lib/x86_64-linux-gnu .. \
  && make -j"$(nproc)" \
  && make -j"$(nproc)" install \
  && ldconfig \
  && cd /tmp \
  && rm -rf vigra

####################################################################
#
# Copy source tree (kept in the image for development)
#
####################################################################
WORKDIR /opt/hm/src/hm
COPY . /opt/hm/src/hm
RUN chown -R "${UID}:${GID}" /opt/hm

####################################################################
#
# Python deps (runtime + build)
#
####################################################################
# Keep this ABI-sensitive stack in lockstep with Torch-TensorRT in
# requirements.txt. These versions intentionally are not build arguments.
RUN pip install --no-cache-dir --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    "torch==2.11.0+cu130" \
    "torchvision==0.26.0+cu130" \
    "torchaudio==2.11.0+cu130"

# PyNvVideoCodec is hosted on NVIDIA's index for many setups.
# Avoid `nvidia-pyindex` here: its build backend expects `pip` to be importable
# inside an isolated PEP517 build env, which isn't always true.
RUN pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com PyNvVideoCodec

RUN pip install --no-cache-dir -r requirements.txt

####################################################################
#
# Build + install native tools and wheels
#
####################################################################
# Build only the Hugin CLI tools used by HockeyMON.
# The upstream CMake install rules assume a full build, so we build the
# needed targets and manually install the resulting binaries + huginbase.
RUN set -eux; \
  HUGIN_BUILD_DIR="/tmp/hugin_cli_build"; \
  rm -rf "${HUGIN_BUILD_DIR}"; \
  cmake -S /opt/hm/src/hm/external/hugin -B "${HUGIN_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DDISABLE_DPKG=ON \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON; \
  cmake --build "${HUGIN_BUILD_DIR}" -- -j"$(nproc)" pto_gen autooptimiser nona; \
  install -m 0755 "${HUGIN_BUILD_DIR}/src/tools/pto_gen" /usr/local/bin/pto_gen; \
  install -m 0755 "${HUGIN_BUILD_DIR}/src/tools/autooptimiser" /usr/local/bin/autooptimiser; \
  install -m 0755 "${HUGIN_BUILD_DIR}/src/tools/nona" /usr/local/bin/nona; \
  mkdir -p /usr/local/lib/hugin; \
  cp -a "${HUGIN_BUILD_DIR}/src/hugin_base/libhuginbase.so"* /usr/local/lib/hugin/; \
  ldconfig

# rules_python's hermetic Python toolchain refuses to run as root.
USER "${USERNAME}"
RUN export HOME="/home/${USERNAME}" \
  && rm -rf "${HOME}/.cache/bazel" \
  && cd /opt/hm/src/hm \
  && rm -rf dist && mkdir -p dist \
  && export CUDA_BAZEL_ARCHS="sm_75;sm_80;sm_86;sm_89;sm_90;compute_120:compute_120,sm_120" \
  && ./bazel/bazel.sh run //hockeymon:bdist_wheel \
  && ./bazel/bazel.sh run //hmlib:bdist_wheel

USER root
RUN cd /opt/hm/src/hm \
  && pip install --no-cache-dir dist/hockeymon-*.whl dist/hmlib-*.whl

####################################################################
#
# Install local Python packages (OpenMMLab + xmodels)
#
####################################################################
RUN set -eux; \
  export CUDA_HOME=/usr/local/cuda; \
  export FORCE_CUDA=1; \
  export MMCV_WITH_OPS=1; \
  export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;12.0+PTX"; \
  export MAX_JOBS="$(nproc)"; \
  WHEEL_DIR="/tmp/local_wheels"; \
  rm -rf "${WHEEL_DIR}"; \
  mkdir -p "${WHEEL_DIR}"; \
  build_install() { \
    local src="$1"; \
    local glob="$2"; \
    pip wheel --no-build-isolation --no-deps -w "${WHEEL_DIR}" "${src}"; \
    pip install --no-cache-dir --no-build-isolation "${WHEEL_DIR}"/${glob}; \
    rm -f "${WHEEL_DIR}"/${glob}; \
  }; \
  \
  build_install /opt/hm/src/hm/openmm/mmengine "mmengine-*.whl"; \
  build_install /opt/hm/src/hm/openmm/mmcv "mmcv-*.whl"; \
  build_install /opt/hm/src/hm/openmm/mmdetection "mmdet-*.whl"; \
  build_install /opt/hm/src/hm/openmm/mmsegmentation "mmsegmentation-*.whl"; \
  build_install /opt/hm/src/hm/openmm/mmpretrain "mmpretrain-*.whl"; \
  build_install /opt/hm/src/hm/openmm/mmyolo "mmyolo-*.whl"; \
  build_install /opt/hm/src/hm/openmm/mmaction2 "mmaction2-*.whl"; \
  if [ -d /opt/hm/src/hm/openmm/mmrotate/src/e2cnn ]; then \
    build_install /opt/hm/src/hm/openmm/mmrotate/src/e2cnn "e2cnn-*.whl"; \
  fi; \
  build_install /opt/hm/src/hm/openmm/mmrotate "mmrotate-*.whl"; \
  build_install /opt/hm/src/hm/openmm/mmocr "mmocr-*.whl"; \
  build_install /opt/hm/src/hm/openmm/mmpose "mmpose-*.whl"; \
  build_install /opt/hm/src/hm/openmm/mmhuman3d "mmhuman3d-*.whl"; \
  build_install /opt/hm/src/hm/openmm/mmeval "mmeval-*.whl"; \
  build_install /opt/hm/src/hm/xmodels/LightGlue "lightglue-*.whl"; \
  build_install /opt/hm/src/hm/xmodels/str/parseq "strhub-*.whl"; \
  \
  rm -rf "${WHEEL_DIR}"

####################################################################
#
# Runtime defaults
#
####################################################################
USER "${USERNAME}"
WORKDIR /home/"${USERNAME}"
RUN mkdir -p "/home/${USERNAME}/Videos"

CMD ["/bin/bash"]
