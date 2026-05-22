#include "sed/trt_yolo_infer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>

namespace sahi {

// ===================================================================
// TrtLogger
// ===================================================================
namespace {
class TrtLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO)
            std::cout << "[TRT] " << msg << std::endl;
    }
};
}  // namespace

// ===================================================================
// Construction / Destruction
// ===================================================================
TrtYoloInfer::TrtYoloInfer() = default;

TrtYoloInfer::~TrtYoloInfer() {
    release();
}

// ===================================================================
// configure() — lightweight, no engine needed
// ===================================================================
void TrtYoloInfer::configure(int input_w, int input_h,
                              int num_anchors, int output_dim,
                              float conf_thresh, float nms_iou) {
    input_w_ = input_w;
    input_h_ = input_h;
    num_anchors_ = num_anchors;
    output_dim_ = output_dim;
    conf_thresh_ = conf_thresh;
    nms_iou_ = nms_iou;
}

// ===================================================================
// init()
// ===================================================================
bool TrtYoloInfer::init(const std::string& engine_path,
                         int input_w, int input_h,
                         int max_batch_size,
                         float conf_thresh, float nms_iou) {
    input_w_ = input_w;
    input_h_ = input_h;
    max_batch_size_ = max_batch_size;
    conf_thresh_ = conf_thresh;
    nms_iou_ = nms_iou;

    // ── 1. 加载 engine 文件 ──
    std::ifstream ef(engine_path, std::ios::binary);
    if (!ef) {
        std::cerr << "[TrtYoloInfer] Cannot open engine: " << engine_path << std::endl;
        return false;
    }
    ef.seekg(0, std::ios::end);
    size_t engine_size = ef.tellg();
    ef.seekg(0, std::ios::beg);
    std::vector<char> engine_data(engine_size);
    ef.read(engine_data.data(), engine_size);
    ef.close();

    // ── 2. Deserialize engine ──
    static TrtLogger logger;
    runtime_ = nvinfer1::createInferRuntime(logger);
    if (!runtime_) {
        std::cerr << "[TrtYoloInfer] Failed to create TRT runtime" << std::endl;
        return false;
    }
    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), engine_size);
    if (!engine_) {
        std::cerr << "[TrtYoloInfer] Failed to deserialize engine" << std::endl;
        delete runtime_;
        runtime_ = nullptr;
        return false;
    }

    // ── 3. Create execution context ──
    context_ = engine_->createExecutionContext();
    if (!context_) {
        std::cerr << "[TrtYoloInfer] Failed to create execution context" << std::endl;
        delete engine_;
        engine_ = nullptr;
        delete runtime_;
        runtime_ = nullptr;
        return false;
    }

    // ── 4. 读取 tensor 信息 + 缓存名称 ──
    int n_io = engine_->getNbIOTensors();
    for (int i = 0; i < n_io; i++) {
        const char* name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            input_name_ = name;
        } else {
            output_name_ = name;
            auto dims = engine_->getTensorShape(name);
            if (dims.nbDims == 3) {
                num_anchors_ = dims.d[1];   // [batch, 8400, 6]
                output_dim_ = dims.d[2];
            }
        }
    }
    std::cout << "[TrtYoloInfer] Engine loaded: " << num_anchors_
              << " anchors, output dim=" << output_dim_
              << ", max_batch=" << max_batch_size_ << std::endl;

    // ── 5. 计算 per-image 元素数 + 分配 GPU buffers ──
    input_per_image_ = 3 * input_w_ * input_h_;
    output_per_image_ = num_anchors_ * output_dim_;
    input_bytes_ = static_cast<size_t>(max_batch_size_) * input_per_image_ * sizeof(float);
    output_bytes_ = static_cast<size_t>(max_batch_size_) * output_per_image_ * sizeof(float);

    if (cudaMalloc(&input_gpu_, input_bytes_) != cudaSuccess) {
        std::cerr << "[TrtYoloInfer] Failed to allocate GPU input buffer" << std::endl;
        delete context_; context_ = nullptr;
        delete engine_;  engine_ = nullptr;
        delete runtime_; runtime_ = nullptr;
        return false;
    }
    if (cudaMalloc(&output_gpu_, output_bytes_) != cudaSuccess) {
        std::cerr << "[TrtYoloInfer] Failed to allocate GPU output buffer" << std::endl;
        cudaFree(input_gpu_); input_gpu_ = nullptr;
        delete context_; context_ = nullptr;
        delete engine_;  engine_ = nullptr;
        delete runtime_; runtime_ = nullptr;
        return false;
    }

    // ── 6. Create CUDA stream ──
    cudaStreamCreate(&stream_);

    return true;
}

// ===================================================================
// preprocess()  — Step 2B
// ===================================================================
PreprocessParams TrtYoloInfer::preprocess(const cv::Mat& bgr,
                                           float* gpu_input,
                                           cudaStream_t stream) {
    int orig_h = bgr.rows;
    int orig_w = bgr.cols;

    // Step 1: scale ratio
    float gain = std::min(static_cast<float>(input_w_) / orig_w,
                          static_cast<float>(input_h_) / orig_h);
    int new_w = static_cast<int>(orig_w * gain);
    int new_h = static_cast<int>(orig_h * gain);

    // Step 2: resize
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // Step 3: letterbox pad to input size
    int pad_w = (input_w_ - new_w) / 2;
    int pad_h = (input_h_ - new_h) / 2;
    cv::Mat letterbox(input_h_, input_w_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(letterbox(cv::Rect(pad_w, pad_h, new_w, new_h)));

    // Step 4: BGR → RGB, float32, /255.0
    cv::Mat rgb;
    cv::cvtColor(letterbox, rgb, cv::COLOR_BGR2RGB);
    cv::Mat rgb_float;
    rgb.convertTo(rgb_float, CV_32FC3, 1.0 / 255.0);

    // Step 5: HWC → CHW, copy to GPU
    std::vector<cv::Mat> channels(3);
    cv::split(rgb_float, channels);
    for (int c = 0; c < 3; c++) {
        cudaMemcpyAsync(gpu_input + c * input_w_ * input_h_,
                        channels[c].data,
                        input_w_ * input_h_ * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
    }

    return {gain, pad_w, pad_h};
}

// ===================================================================
// infer()  — Step 2D
// ===================================================================
std::vector<YoloBox> TrtYoloInfer::infer(const cv::Mat& bgr) {
    auto results = infer_batch({bgr});
    if (results.empty()) return {};
    return results[0];
}

// ===================================================================
// infer_batch()  — Step 2D
// ===================================================================
std::vector<std::vector<YoloBox>> TrtYoloInfer::infer_batch(
    const std::vector<cv::Mat>& bgrs) {

    int n = static_cast<int>(bgrs.size());
    if (n == 0) return {};
    if (n > max_batch_size_) {
        std::cerr << "[TrtYoloInfer] Batch size " << n
                  << " exceeds max " << max_batch_size_ << std::endl;
        return {};
    }

    // ── 1. Preprocess each image → GPU input buffer ──
    std::vector<PreprocessParams> pp_list(n);
    std::vector<int> orig_w_list(n), orig_h_list(n);

    for (int i = 0; i < n; i++) {
        orig_w_list[i] = bgrs[i].cols;
        orig_h_list[i] = bgrs[i].rows;
        float* gpu_offset = input_gpu_ + i * input_per_image_;
        pp_list[i] = preprocess(bgrs[i], gpu_offset, stream_);
    }

    // ── 2. Set dynamic batch dimension ──
    nvinfer1::Dims4 input_shape{n, 3, input_h_, input_w_};
    if (!context_->setInputShape(input_name_.c_str(), input_shape)) {
        std::cerr << "[TrtYoloInfer] setInputShape failed for batch=" << n << std::endl;
        return {};
    }

    // ── 3. Set tensor addresses + enqueue ──
    context_->setTensorAddress(input_name_.c_str(), input_gpu_);
    context_->setTensorAddress(output_name_.c_str(), output_gpu_);

    if (!context_->enqueueV3(stream_)) {
        std::cerr << "[TrtYoloInfer] enqueueV3 failed" << std::endl;
        return {};
    }
    cudaStreamSynchronize(stream_);

    // ── 4. Download output from GPU → CPU ──
    size_t result_floats = static_cast<size_t>(n) * output_per_image_;
    std::vector<float> cpu_output(result_floats);
    cudaMemcpy(cpu_output.data(), output_gpu_,
               result_floats * sizeof(float),
               cudaMemcpyDeviceToHost);

    // ── 5. Decode each image independently ──
    std::vector<std::vector<YoloBox>> results(n);
    for (int i = 0; i < n; i++) {
        const float* img_output = cpu_output.data() + i * output_per_image_;
        results[i] = decode_outputs(img_output, pp_list[i],
                                     orig_w_list[i], orig_h_list[i]);
    }

    return results;
}

// ===================================================================
// decode_outputs()  — Step 2C
// ===================================================================
std::vector<YoloBox> TrtYoloInfer::decode_outputs(
    const float* raw_output,
    const PreprocessParams& pp,
    int orig_w, int orig_h) const {

    std::vector<YoloBox> boxes;

    for (int a = 0; a < num_anchors_; a++) {
        const float* det = raw_output + a * output_dim_;  // [x1,y1,x2,y2,conf,cls]

        float x1 = det[0];
        float y1 = det[1];
        float x2 = det[2];
        float y2 = det[3];
        float conf = det[4];
        int cls_id = static_cast<int>(det[5]);

        if (conf < conf_thresh_) continue;
        if (cls_id < 0) continue;

        // Undo letterbox: (x_net - pad) / gain
        float inv_gain = 1.0f / pp.gain;
        x1 = (x1 - pp.pad_w) * inv_gain;
        y1 = (y1 - pp.pad_h) * inv_gain;
        x2 = (x2 - pp.pad_w) * inv_gain;
        y2 = (y2 - pp.pad_h) * inv_gain;

        // Clamp to image bounds
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(orig_w)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(orig_h)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(orig_w)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(orig_h)));

        if (x2 <= x1 || y2 <= y1) continue;

        boxes.push_back({x1, y1, x2, y2, conf, cls_id});
    }

    // Apply per-class NMS
    return nms_per_class(std::move(boxes), nms_iou_);
}

// ===================================================================
// nms_per_class()
// ===================================================================
std::vector<YoloBox> TrtYoloInfer::nms_per_class(
    std::vector<YoloBox> boxes, float iou_thresh) {

    if (boxes.empty()) return {};

    // Group by class_id
    std::map<int, std::vector<size_t>> class_indices;
    for (size_t i = 0; i < boxes.size(); i++) {
        class_indices[boxes[i].class_id].push_back(i);
    }

    std::vector<bool> keep(boxes.size(), true);

    for (auto& kv : class_indices) {
        auto& indices = kv.second;

        // Sort by confidence descending
        std::sort(indices.begin(), indices.end(),
                  [&](size_t a, size_t b) {
                      return boxes[a].confidence > boxes[b].confidence;
                  });

        for (size_t i = 0; i < indices.size(); i++) {
            size_t idx_i = indices[i];
            if (!keep[idx_i]) continue;

            float xi1 = boxes[idx_i].x1;
            float yi1 = boxes[idx_i].y1;
            float xi2 = boxes[idx_i].x2;
            float yi2 = boxes[idx_i].y2;
            float area_i = (xi2 - xi1) * (yi2 - yi1);

            for (size_t j = i + 1; j < indices.size(); j++) {
                size_t idx_j = indices[j];
                if (!keep[idx_j]) continue;

                float xj1 = boxes[idx_j].x1;
                float yj1 = boxes[idx_j].y1;
                float xj2 = boxes[idx_j].x2;
                float yj2 = boxes[idx_j].y2;

                float inter_w = std::max(0.0f, std::min(xi2, xj2) - std::max(xi1, xj1));
                float inter_h = std::max(0.0f, std::min(yi2, yj2) - std::max(yi1, yj1));
                float inter = inter_w * inter_h;
                if (inter <= 0.0f) continue;

                float area_j = (xj2 - xj1) * (yj2 - yj1);
                float iou = inter / (area_i + area_j - inter);

                if (iou > iou_thresh) {
                    keep[idx_j] = false;
                }
            }
        }
    }

    std::vector<YoloBox> result;
    for (size_t i = 0; i < boxes.size(); i++) {
        if (keep[i]) result.push_back(boxes[i]);
    }
    return result;
}

// ===================================================================
// release()
// ===================================================================
void TrtYoloInfer::release() {
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    if (input_gpu_) {
        cudaFree(input_gpu_);
        input_gpu_ = nullptr;
    }
    if (output_gpu_) {
        cudaFree(output_gpu_);
        output_gpu_ = nullptr;
    }
    if (context_) {
        delete context_;
        context_ = nullptr;
    }
    if (engine_) {
        delete engine_;
        engine_ = nullptr;
    }
    if (runtime_) {
        delete runtime_;
        runtime_ = nullptr;
    }
}

}  // namespace sahi
