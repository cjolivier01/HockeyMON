#
# HockeyMON Top-Level CMake Configuration File
#
cmake_minimum_required(VERSION 3.8)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/external/hugin/CMakeModules")

project(hockeymon)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(BUILD_RELEASE_FLAGS "-g -O3")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BUILD_RELEASE_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${BUILD_RELEASE_FLAGS}")
elseif(CMAKE_BUILD_TYPE STREQUAL "ReleaseWithDebugInfo")
  message(STATUS "Building in ReleaseWithDebugInfo mode.")
  set(BUILD_PROFILE_FLAGS "-fno-omit-frame-pointer -g -O1")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BUILD_PROFILE_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${BUILD_PROFILE_FLAGS}")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(BUILD_DEBUG_FLAGS "-fno-omit-frame-pointer -O0 -g")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BUILD_DEBUG_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${BUILD_DEBUG_FLAGS}")

  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHUGIN_DEBUG=1")
  # include_directories(${CMAKE_CURRENT_LIST_DIR}/../vigra/include)
  # link_directories(${CMAKE_CURRENT_LIST_DIR}/../vigra/build/src/impex)

  # set(VIGRA_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../vigra/include)
  # set(VIGRA_LIBRARIES ${CMAKE_CURRENT_LIST_DIR}/../vigra/build/src/impex/libvigraimpex.so.11)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)

set(WITH_OPENCV 1)

# set(CMAKE_CXX_STANDARD 14)

# OpenCV config
SET(BUILD_opencv_apps OFF)

# SET(BUILD_opencv_freetype OFF)
# SET(BUILD_SHARED_LIBS OFF)
SET(BUILD_TESTS OFF)
SET(BUILD_PERF_TESTS OFF)
SET(BUILD_DOCS OFF)

# SET(PYTHON3_LIMITED_API ON)
SET(BUILD_OPENEXR ON)

# Mine
SET(CUDA_NVCC_FLAGS "-Xcompiler,\"-std=c++14\"")

# SET(CMAKE_C_COMPILER gcc)
# SET(CMAKE_CXX_COMPILER g++)
# SET(INSTALL_PYTHON_EXAMPLES OFF)
# SET(INSTALL_C_EXAMPLES OFF)
# SET(EIGEN3_INCLUDE_PATH /usr/include/eigen3)
SET(OPENCV_ENABLE_NONFREE ON)

# SET(WITH_CUDA ON)
# SET(WITH_CUDNN ON)
# SET(OPENCV_DNN_CUDA ON)
SET(ENABLE_FAST_MATH 1)

# SET(CUDA_FAST_MATH 1)
# SET(CUDA_ARCH_BIN 6.1)
# SET(WITH_CUBLAS 1)
# SET(OPENCV_DNN_CUDA 0)

# SET(WITH_EIGEN 0)
# SET(HAVE_opencv_python3 ON)
SET(BUILD_EXAMPLES OFF)

# SET(OPENCV_EXTRA_MODULES_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/opencv-python/opencv_contrib/modules")
# SET(OPENCV_EXTRA_MODULES_PATH "${CMAKE_CURRENT_SOURCE_DIR}/opencv_contrib/modules")
# MESSAGE(FATAL " OPENCV_EXTRA_MODULES_PATH=${OPENCV_EXTRA_MODULES_PATH}")
# SET(OPENCV_EXTRA_MODULES_PATH "${CMAKE_CURRENT_LIST_DIR}/external/opencv-python/opencv_contrib/modules/")

# include_directories(${CMAKE_CURRENT_LIST_DIR}/external/opencv-python/opencv/include/opencv2)
# Detect compiler version (does not seem to work)
# if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang|MSVC")
# # Using regular expression to extract major version
# string(REGEX MATCH "^[0-9]+" COMPILER_MAJOR_VERSION "${CMAKE_CXX_COMPILER_VERSION}")
# message(STATUS "Compiler major version: ${COMPILER_MAJOR_VERSION}")
# set(CMAKE_CXX_FLAGS "-isystem /usr/include/c++/${COMPILER_MAJOR_VERSION} ${CMAKE_CXX_FLAGS}")
# else()
# message(WARNING "Compiler not recognized. Version detection might not be accurate.")
# endif()

# set(CMAKE_CXX_FLAGS "-isystem /usr/include/c++/${COMPILER_MAJOR_VERSION} ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fdiagnostics-color=always")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=leak")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

if(HM_BUILD_ASAN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fsanitize=address -fno-omit-frame-pointer -UDEBUG")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -UDEBUG")
endif(HM_BUILD_ASAN)

# link_directories($ENV{CONDA_PREFIX}/x86_64-conda-linux-gnu/sysroot/usr/lib64)
find_package(Torch REQUIRED)

# message(FATAL_ERROR "TORCH_INCLUDE_DIRS=${TORCH_INCLUDE_DIRS}")
include_directories(${TORCH_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(TORCH_LIBRARY_PATH "${TORCH_INSTALL_PREFIX}/lib")
link_directories(${TORCH_LIBRARY_PATH})

set(THREADS_PREFER_PTHREAD_FLAG 0)

# set(Threads_FOUND 1)
# add_compile_options(-pthread)
find_package(Threads REQUIRED)
message(STATUS " CMAKE_THREAD_LIBS_INIT=${CMAKE_THREAD_LIBS_INIT}")
set(CMAKE_THREAD_LIBS_INIT "-lpthread")

link_directories($ENV{CONDA_PREFIX}/lib)

#
# FFMpeg
#
find_package(PkgConfig REQUIRED)

pkg_check_modules(AVCODEC REQUIRED libavcodec)
pkg_check_modules(AVFORMAT REQUIRED libavformat)
pkg_check_modules(AVUTIL REQUIRED libavutil)
pkg_check_modules(SWSCALE REQUIRED libswscale)

# Include FFmpeg headers
include_directories(${AVCODEC_INCLUDE_DIRS})
include_directories(${AVFORMAT_INCLUDE_DIRS})
include_directories(${AVUTIL_INCLUDE_DIRS})
include_directories(${SWSCALE_INCLUDE_DIRS})

set(FFMPEG_LIBS ${AVCODEC_LIBRARIES} ${AVFORMAT_LIBRARIES} ${AVFORMAT_LIBRARIES} ${AVUTIL_LIBRARIES} ${SWSCALE_LIBRARIES})

if (WITH_OPENCV)
  # cudaimgproc
  find_package(OpenCV REQUIRED core calib3d highgui imgcodecs imgproc)
  set(OpenCV_ENABLE_OPENGL TRUE)  # Print OpenCV version and directories (optional but helpful for debugging)
  message(STATUS "OpenCV version: ${OpenCV_VERSION}")
  message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
  message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
  add_compile_options(-DWITH_OPENCV=1)
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# message(FATAL_ERROR " FFMPEG_LIBS=${FFMPEG_LIBS}")

#
# FFMpeg-
#
#include_directories(/usr/include/opencv4)
# find_package(
# Threads REQUIRED
# PATHS
# $ENV{CONDA_PREFIX}
# $ENV{CONDA_PREFIX}/x86_64-conda-linux-gnu/sysroot/usr
# )
# message(FATAL_ERROR "Stop here")

#
#
# OPTIMIZATIONS TURNED OFF!!
#
#
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -fno-omit-frame-pointer")
find_package(PythonInterp REQUIRED)
find_package(PythonLibs "${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}" REQUIRED)

if(PYTHONLIBS_FOUND)
  include_directories(${PYTHON_INCLUDE_DIRS})
  message("Using default python version: " ${PYTHONLIBS_VERSION_STRING})
endif(PYTHONLIBS_FOUND)

# Find required protobuf package
# find_package(Protobuf REQUIRED)
set(HAS_UNINSTALL_TARGET 1)

set(HOCKEYMON_TOP_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(HOCKEYMON_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(external/pybind11)
include_directories(SYSTEM external/pybind11/include)
include_directories(SYSTEM external/eigen)
include_directories(SYSTEM external/eigen/unsupported)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/abseil-cpp)

# Very top dir is include dir start for project-wide relative imports
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# And the external dir next
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "i686|x86|i386")
  # Include hugin sources as include dirs
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/hugin/src/hugin_base)
endif()

find_package(OpenGL REQUIRED
  PATHS /home/colivier/.conda/envs/ubuntu-gpu
)
include_directories(${OPENGL_INCLUDE_DIR})

FIND_PACKAGE(FLANN REQUIRED)
SET(FLANN_FOUND 1)

add_subdirectory(external)

# add_subdirectory(external/eigen)
# add_subdirectory(external/abseil-cpp)
set(OPENCV_EXTRA_MODULES_PATH ${CMAKE_CURRENT_SOURCE_DIR}/external/opencv-python/opencv_contrib/modules)
set(BUILD_SHARED_LIBS OFF)

# add_subdirectory(external/opencv-python/opencv)

# add_subdirectory(external/hugin)
add_subdirectory(hockeymon)
