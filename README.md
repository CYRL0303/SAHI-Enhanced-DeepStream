# SAHI Enhanced DeepStream

Standalone **DeepStream + SAHI bypass** inference framework for edge devices.

This project extracts the SAHI (Slicing Aided Hyper Inference) integration pattern from a ship assistant driving system into a self-contained C++ library. It runs a GStreamer pipeline with dual-branch inference — DeepStream's built-in `nvinfer` for full-image detection at low resolution, plus a parallel high-resolution SAHI path using a standalone TensorRT engine for small-object detection — and merges both results via per-class NMS.

No ROS2 dependency. Drop it into any Jetson or NVIDIA edge device.

## Architecture

```
RTSP Camera ──► tee ──► 640x480 NV12  ──► nvstreammux ──► nvinfer ──► fakesink
                 │                                                    │
                 │ (1920x1080 BGRx)                                   │ pad probe
                 ▼                                                    │
            appsink                                                   │
                 │                                                    │
                 ▼                                                    ▼
         Frame Queue (depth=2)                              full-image dets
                 │                                                    │
                 ▼                                                    │
          SAHI Thread                                                  │
          ├─ SahiSlicer::compute_slices()                              │
          ├─ TrtYoloInfer::infer_batch()  (batch TRT)                  │
          └─ coordinate translation (slice → full-image)               │
                 │                                                    │
                 ▼                                                    ▼
          slice dets  ──────►  sahi_nms_merge()  ◄──────  full dets
                                        │
                                        ▼
                                 DetectionCallback
                                 (user-supplied)
```

**Key design decisions:**

| Decision | Why |
|----------|-----|
| Dual resolution | DeepStream at 640×480 for speed; SAHI at 1920×1080 for small-object recall |
| Dedicated SAHI thread | TRT inference latency never blocks the GStreamer pipeline |
| Frame queue depth 2 | Prevents queue buildup when SAHI inference is slower than camera FPS |
| 500ms staleness check | Keeps SAHI results temporally aligned with DeepStream full-image dets |
| Concatenation + NMS merge | No source weighting — the NMS algorithm naturally picks the highest-confidence box |

## Prerequisites

### Hardware

- **NVIDIA Jetson** (Orin AGX / Orin NX / Orin Nano / Xavier NX / Xavier AGX)
- Or any **x86_64 + NVIDIA dGPU** running JetPack-compatible software (see [DeepStream system requirements](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html))

### Software

| Component | Version | Where to get it |
|-----------|---------|-----------------|
| JetPack | 6.0+ (Jetson Orin) or 5.x (Xavier) | [NVIDIA SDK Manager](https://developer.nvidia.com/sdk-manager) |
| DeepStream SDK | 7.0 / 7.1 | `sudo apt install deepstream-7.1` (included in JetPack 6) |
| TensorRT | 8.6+ | Bundled with JetPack under `/usr/lib/aarch64-linux-gnu` |
| CUDA | 11.4+ / 12.x | Bundled with JetPack under `/usr/local/cuda` |
| OpenCV | 4.5+ (with GStreamer) | `sudo apt install libopencv-dev` |
| GStreamer | 1.20+ | Pre-installed on JetPack |
| CMake | 3.8+ | `sudo apt install cmake` |
| C++17 compiler | g++ 9+ | `sudo apt install build-essential` |

## Directory Structure

```
SAHI Enhanced DeepStream/
├── CMakeLists.txt                    # Build system
├── README.md
├── include/sed/
│   ├── sahi_slicer.hpp               # Slice coordinate engine
│   ├── sahi_nms.hpp                  # Dual-source NMS merge
│   ├── trt_yolo_infer.hpp            # TensorRT YOLO inference
│   └── deepstream_sahi.hpp           # Main framework class
├── src/
│   ├── sahi_slicer.cpp
│   ├── sahi_nms.cpp
│   ├── trt_yolo_infer.cpp
│   ├── deepstream_sahi.cpp           # Framework implementation
│   └── main.cpp                      # Demo application
├── config/
│   ├── deepstream_config_yolo.txt    # DeepStream nvinfer config
│   ├── labels.txt                    # 80-class COCO labels
│   └── tracker_config.txt            # (unused, reserved)
├── tools/
│   └── build_batch_engine.py         # Build TRT FP16 dynamic-batch engine from ONNX
└── test/
    ├── test_sahi_slicer.cpp          # Slice engine unit tests (pure C++)
    ├── test_nms.cpp                  # NMS merge verification
    ├── trt_introspect.cpp            # TRT engine inspector
    ├── bench_latency.cpp             # Inference latency benchmark
    ├── viz_nms_merge.py              # NMS merge visualization
    └── verify_nms_merge.py           # Cross-validation C++ vs Python NMS
```

## Quick Start

### 1. Clone to Jetson

```bash
git clone <your-repo-url> SAHI Enhanced DeepStream
cd SAHI Enhanced DeepStream
```

### 2. Build the TRT engine

You need a YOLOv8n ONNX model (or any YOLOv8 variant with post-processed output `[batch, 8400, 6]`):

```bash
# Install Python deps
pip install onnx onnxruntime ultralytics

# Build FP16 engine with dynamic batch support
python3 tools/build_batch_engine.py \
    --onnx /path/to/yolov8n.onnx \
    --engine config/yolov8n_batch.engine \
    --max-batch 8
```

> **What engine format is needed?** The TRT engine must output `[batch, 8400, 6]` where the 6 values are `[x1, y1, x2, y2, confidence, class_id]` in network-space coordinates (not raw 84-class logits). The provided `build_batch_engine.py` handles this. See `test/trt_introspect.cpp` to inspect an existing engine.

### 3. Configure DeepStream nvinfer

Edit `config/deepstream_config_yolo.txt` to point to your model and label files:

```ini
[property]
model-engine-file=/absolute/path/to/SAHI Enhanced DeepStream/config/yolov8n_batch.engine
labelfile-path=/absolute/path/to/SAHI Enhanced DeepStream/config/labels.txt
custom-lib-path=/path/to/libnvdsinfer_custom_impl_Yolo.so
# ... other settings usually fine as-is
```

> `libnvdsinfer_custom_impl_Yolo.so` is the DeepStream YOLO custom parser. You can build it from [NVIDIA's deepstream_yolo repo](https://github.com/NVIDIA-AI-IOT/deepstream_yolo) or use the one bundled with your JetPack installation.

### 4. Build

```bash
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON
make -j$(nproc)
```

This produces:
- `libsed_core.a` — reusable static library
- `sed_demo` — demo application
- Test executables (if `BUILD_TESTS=ON`)

### 5. Configure the demo

Edit `src/main.cpp` and adjust the `DsSahiConfig` fields for your environment:

```cpp
cfg.rtsp_url1 = "rtsp://your-camera-1:8554/live/stream";
cfg.rtsp_url2 = "rtsp://your-camera-2:8554/live/stream";
cfg.enable_sahi = true;
cfg.sahi_engine_path  = "/path/to/config/yolov8n_batch.engine";
cfg.class_labels_path = "/path/to/config/labels.txt";
cfg.nvinfer_config_path = "/path/to/config/deepstream_config_yolo.txt";
```

### 6. Run

```bash
./sed_demo
```

Press `Ctrl+C` to stop gracefully.

## Configuration Reference

All options live in `sed::DsSahiConfig`:

### Camera

| Field | Default | Description |
|-------|---------|-------------|
| `rtsp_url1` | — | RTSP URL for camera 0 |
| `rtsp_url2` | — | RTSP URL for camera 1 (set same as url1 for single camera) |

### DeepStream inference path

| Field | Default | Description |
|-------|---------|-------------|
| `stream_width` | 640 | Resolution for `nvinfer` full-image inference |
| `stream_height` | 480 | |
| `nvinfer_config_path` | — | Path to `deepstream_config_yolo.txt` |

### SAHI frame capture path

| Field | Default | Description |
|-------|---------|-------------|
| `sahi_frame_width` | 1920 | Capture resolution for SAHI (matches camera native res) |
| `sahi_frame_height` | 1080 | |

### SAHI engine & model

| Field | Default | Description |
|-------|---------|-------------|
| `enable_sahi` | `false` | Master switch; falls back to full-image-only if engine init fails |
| `sahi_engine_path` | — | Path to `.engine` file |
| `class_labels_path` | — | One class name per line |

### SAHI tuning

| Field | Default | Description |
|-------|---------|-------------|
| `slice_batch_size` | 4 | Max slices fed to TRT per `enqueueV3` call |
| `sahi_conf_thresh` | 0.25 | Confidence filter for slice detections |
| `sahi_nms_iou` | 0.45 | Per-class NMS IoU within SAHI decode |
| `sahi_merge_iou` | 0.5 | IoU threshold when merging full + slice detections |
| `sahi_slice_h` / `sahi_slice_w` | 640 × 640 | Pixel dimensions of each slice |
| `sahi_overlap_h` / `sahi_overlap_w` | 0.2 | Overlap ratio (0.2 = 128px for 640 slice) |
| `sahi_result_max_age_ms` | 500 | Staleness threshold for SAHI results |

## Integration Guide

The `main.cpp` demo is intentionally minimal. To embed the framework in your own application:

```cpp
#include "sed/deepstream_sahi.hpp"

int main() {
    sed::DsSahiConfig cfg;
    // ... fill cfg ...

    sed::DeepStreamSahi app(cfg);

    // Register your output handler
    app.set_callback([](const sed::FrameResult &r) {
        for (auto &obj : r.objects) {
            // obj.x1, obj.y1, obj.x2, obj.y2   — bbox in full-image pixels
            // obj.confidence                    — [0, 1]
            // obj.class_id, obj.class_name      — COCO class
            // obj.track_id                      — from DeepStream tracker (-1 if unused)
            // r.camera_id                       — 0 or 1
            // r.frame_num                       — GStreamer frame counter
        }
        // Publish to your message bus, write to DB, trigger downstream logic...
    });

    // Blocks until SIGINT or error
    app.start();
    return 0;
}
```

Link against `libsed_core.a` and add `src/deepstream_sahi.cpp` to your build.

## Running Tests

```bash
# Standalone slicer test (no GPU needed)
./test_sahi_slicer

# NMS merge test (needs TRT libs for YoloBox type, no GPU inference)
./test_nms

# TRT engine inspector
./trt_introspect /path/to/engine.engine

# Latency benchmark
./bench_latency /path/to/engine.engine /path/to/test_image.jpg 10 50
```

## Deploying to Any NVIDIA Edge Device

### Jetson (Orin / Xavier / TX2)

The project is tested on Jetson Orin with JetPack 6.0 and DeepStream 7.1. For other Jetson models:

| Device | JetPack | DeepStream | Notes |
|--------|---------|------------|-------|
| Orin AGX / NX / Nano | 6.0+ | 7.0 / 7.1 | Full support |
| Xavier AGX / NX | 5.1+ | 6.3 / 6.4 | Change `DEEPSTREAM_PATH` in CMakeLists.txt |
| TX2 / TX1 | 4.6 | 6.0 | DeepStream limited; SAHI TRT engine still works standalone |

**Deployment checklist for any Jetson:**

```bash
# 1. Verify JetPack
cat /etc/nv_tegra_release

# 2. Verify DeepStream
deepstream-app --version
ls /opt/nvidia/deepstream/deepstream-*/

# 3. Verify TensorRT
/usr/src/tensorrt/bin/trtexec --version

# 4. Adjust CMakeLists.txt paths if needed
#    - DEEPSTREAM_PATH
#    - CUDA / TRT library search paths

# 5. Build the TRT engine ON the target device
#    (engines are hardware-specific — don't cross-compile them)
python3 tools/build_batch_engine.py --onnx model.onnx --engine model.engine

# 6. cmake + make + run
```

### x86_64 + NVIDIA GPU (Desktop / Server Edge)

The SAHI core (`sed_core`) compiles on any CUDA host. The `deepstream_sahi.cpp` module requires DeepStream SDK (Linux x86_64 only).

```bash
# Install DeepStream for x86_64
# https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html

# Adjust CMakeLists.txt:
#   - Remove aarch64 paths
#   - Set DEEPSTREAM_PATH to /opt/nvidia/deepstream/deepstream-7.1 (x86)
#   - Change /usr/lib/aarch64-linux-gnu → /usr/lib/x86_64-linux-gnu
#   - Change /usr/local/cuda/lib64 (likely still correct)
```

### Without DeepStream (SAHI + TRT only)

If you only need the SAHI slicing + TRT inference (no GStreamer/DeepStream pipeline), compile just the core library:

```bash
# sed_core alone needs: CUDA, TensorRT, OpenCV, pthread
g++ -std=c++17 -Iinclude -I/usr/local/cuda/include \
    src/sahi_slicer.cpp src/sahi_nms.cpp src/trt_yolo_infer.cpp \
    your_app.cpp \
    -lnvinfer -lnvinfer_plugin -lcudart \
    $(pkg-config --libs opencv4) -lpthread -ldl \
    -o your_app
```

This gives you a pure C++ SAHI engine — slice + batch-infer + NMS on any CUDA-capable device, including discrete GPU edge servers.

## Common Issues

### `nvinfer` plugin fails to load
Check that `custom-lib-path` in `deepstream_config_yolo.txt` points to a valid `libnvdsinfer_custom_impl_Yolo.so`. Build it from [deepstream_yolo](https://github.com/NVIDIA-AI-IOT/deepstream_yolo) if needed.

### TRT engine fails to load
The engine must be built **on the same hardware** it runs on. TRT engines are not portable across GPU architectures. Always run `build_batch_engine.py` on the target device.

### SAHI results are stale (age > 500ms)
SAHI inference is slower than the camera frame rate. Increase `slice_batch_size` to amortize TRT overhead, or reduce `sahi_frame_width/height`.

### No RTSP source
For testing without cameras, replace the `rtspsrc` pipeline section with a file source:
```bash
# In create_pipeline(), replace rtspsrc with:
# multifilesrc location=test_%04d.jpg loop=true ! jpegdec ! ...
```

### Memory / performance on low-spec devices
- Reduce `sahi_slice_h/w` (e.g. 480×480)
- Reduce `sahi_frame_width/height` (e.g. 1280×720)
- Increase overlap (fewer slices, less TRT work)
- Set `slice_batch_size=1` (lower GPU memory)
- Use INT8 engine instead of FP16

## License

[Your license here]
