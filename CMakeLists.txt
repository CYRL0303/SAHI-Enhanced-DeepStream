cmake_minimum_required(VERSION 3.8)
project(sed LANGUAGES CXX CUDA)

# ===================================================================
# SED: Standalone DeepStream + SAHI Bypass Framework
#
# Targets Jetson Orin with:
#   - DeepStream 7.1 SDK  (/opt/nvidia/deepstream/deepstream-7.1)
#   - TensorRT + CUDA     (/usr/local/cuda, /usr/lib/aarch64-linux-gnu)
#   - OpenCV              (Jetson default or user-supplied)
#   - GStreamer 1.0
# ===================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------------------------------------------------------------------
# Dependencies
# -------------------------------------------------------------------
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-app-1.0)

# DeepStream 7.1
set(DEEPSTREAM_PATH "/opt/nvidia/deepstream/deepstream-7.1")
if(NOT EXISTS ${DEEPSTREAM_PATH})
    message(FATAL_ERROR "DeepStream SDK not found at ${DEEPSTREAM_PATH}")
endif()

# TensorRT + CUDA  (required — this is a Jetson framework)
if(NOT EXISTS /usr/local/cuda/include)
    message(FATAL_ERROR "CUDA not found at /usr/local/cuda")
endif()
find_library(NVINFER_LIB    nvinfer
    PATHS /usr/lib/aarch64-linux-gnu ${DEEPSTREAM_PATH}/lib
    NO_DEFAULT_PATH REQUIRED)
find_library(NVINFER_PLUGIN nvinfer_plugin
    PATHS /usr/lib/aarch64-linux-gnu ${DEEPSTREAM_PATH}/lib
    NO_DEFAULT_PATH REQUIRED)
find_library(CUDART_LIB     cudart
    PATHS /usr/local/cuda/lib64
    NO_DEFAULT_PATH REQUIRED)

# DeepStream meta libraries
find_library(NVDS_META        nvds_meta
    PATHS ${DEEPSTREAM_PATH}/lib NO_DEFAULT_PATH REQUIRED)
find_library(NVDSGST_META     nvdsgst_meta
    PATHS ${DEEPSTREAM_PATH}/lib NO_DEFAULT_PATH REQUIRED)
find_library(NVDSGST_HELPER   nvdsgst_helper
    PATHS ${DEEPSTREAM_PATH}/lib NO_DEFAULT_PATH REQUIRED)

# -------------------------------------------------------------------
# Include directories
# -------------------------------------------------------------------
include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${GST_INCLUDE_DIRS}
    /usr/include/gstreamer-1.0
    /usr/include/glib-2.0
    /usr/lib/aarch64-linux-gnu/glib-2.0/include
    /usr/local/cuda/include
    ${DEEPSTREAM_PATH}/sources/includes
)

# -------------------------------------------------------------------
# Library  (optional: build SAHI core as a static lib for reuse)
# -------------------------------------------------------------------
set(SAHI_CORE_SOURCES
    src/sahi_slicer.cpp
    src/sahi_nms.cpp
    src/trt_yolo_infer.cpp
)

add_library(sed_core STATIC ${SAHI_CORE_SOURCES})
target_include_directories(sed_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    /usr/local/cuda/include
)
target_link_directories(sed_core PUBLIC
    /usr/local/cuda/lib64
    /usr/lib/aarch64-linux-gnu
    ${DEEPSTREAM_PATH}/lib
)
target_link_libraries(sed_core
    ${NVINFER_LIB} ${NVINFER_PLUGIN} ${CUDART_LIB}
    ${OpenCV_LIBS}
    pthread dl
)

# -------------------------------------------------------------------
# Main executable: sed_demo
# -------------------------------------------------------------------
add_executable(sed_demo
    src/main.cpp
    src/deepstream_sahi.cpp
)

target_link_libraries(sed_demo
    sed_core
    ${OpenCV_LIBS}
    ${GST_LIBRARIES}
    glib-2.0 gobject-2.0 gmodule-2.0
    ${NVDS_META} ${NVDSGST_META} ${NVDSGST_HELPER}
    ${NVINFER_LIB} ${NVINFER_PLUGIN} ${CUDART_LIB}
    pthread dl
)

# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------
option(BUILD_TESTS "Build test programs" ON)

if(BUILD_TESTS)
    # test_sahi_slicer  (pure C++, no CUDA needed)
    add_executable(test_sahi_slicer
        test/test_sahi_slicer.cpp
    )
    target_include_directories(test_sahi_slicer PRIVATE
        include
    )
    target_link_libraries(test_sahi_slicer
        sed_core
        ${OpenCV_LIBS}
        pthread
    )

    # test_nms  (needs trt_yolo_infer_obj)
    add_executable(test_nms
        test/test_nms.cpp
    )
    target_include_directories(test_nms PRIVATE
        include
        /usr/local/cuda/include
        ${OpenCV_INCLUDE_DIRS}
    )
    target_link_directories(test_nms PRIVATE
        /usr/local/cuda/lib64
        /usr/lib/aarch64-linux-gnu
        ${DEEPSTREAM_PATH}/lib
    )
    target_link_libraries(test_nms
        sed_core
        ${NVINFER_LIB} ${NVINFER_PLUGIN} ${CUDART_LIB}
        ${OpenCV_LIBS}
        pthread dl
    )

    # trt_introspect
    add_executable(trt_introspect
        test/trt_introspect.cpp
    )
    target_include_directories(trt_introspect PRIVATE
        include
        /usr/local/cuda/include
        ${OpenCV_INCLUDE_DIRS}
    )
    target_link_directories(trt_introspect PRIVATE
        /usr/local/cuda/lib64
        /usr/lib/aarch64-linux-gnu
        ${DEEPSTREAM_PATH}/lib
    )
    target_link_libraries(trt_introspect
        sed_core
        ${NVINFER_LIB} ${NVINFER_PLUGIN} ${CUDART_LIB}
        ${OpenCV_LIBS}
        pthread dl
    )

    # bench_latency
    add_executable(bench_latency
        test/bench_latency.cpp
    )
    target_include_directories(bench_latency PRIVATE
        include
        /usr/local/cuda/include
        ${OpenCV_INCLUDE_DIRS}
    )
    target_link_directories(bench_latency PRIVATE
        /usr/local/cuda/lib64
        /usr/lib/aarch64-linux-gnu
        ${DEEPSTREAM_PATH}/lib
    )
    target_link_libraries(bench_latency
        sed_core
        ${NVINFER_LIB} ${NVINFER_PLUGIN} ${CUDART_LIB}
        ${OpenCV_LIBS}
        pthread dl
    )

    message(STATUS "Tests: ENABLED")
endif()

message(STATUS "SED framework configured")
