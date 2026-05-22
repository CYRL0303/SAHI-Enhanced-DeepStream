#pragma once

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

namespace sahi {

struct YoloBox {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
};

struct PreprocessParams {
    float gain;  // scale ratio = min(input_w / orig_w, input_h / orig_h)
    int pad_w;   // horizontal letterbox padding (pixels on each side)
    int pad_h;   // vertical letterbox padding
};

class TrtYoloInfer {
public:
    TrtYoloInfer();
    ~TrtYoloInfer();

    TrtYoloInfer(const TrtYoloInfer&) = delete;
    TrtYoloInfer& operator=(const TrtYoloInfer&) = delete;

    // Full init: load engine + allocate GPU buffers (for inference)
    bool init(const std::string& engine_path,
              int input_w, int input_h,
              int max_batch_size,
              float conf_thresh, float nms_iou);

    // Lightweight configure: set params only (for decode-only use)
    void configure(int input_w, int input_h,
                   int num_anchors, int output_dim,
                   float conf_thresh, float nms_iou);

    // ── Inference ──
    // Single-image inference: preprocess → TRT enqueue → decode + NMS
    std::vector<YoloBox> infer(const cv::Mat& bgr);

    // Batch inference: N images → one TRT enqueue → per-image decode + NMS
    // Each image's original dimensions are taken from bgr.cols / bgr.rows
    std::vector<std::vector<YoloBox>> infer_batch(
        const std::vector<cv::Mat>& bgrs);

    PreprocessParams preprocess(const cv::Mat& bgr,
                                float* gpu_input,
                                cudaStream_t stream);

    std::vector<YoloBox> decode_outputs(const float* raw_output,
                                        const PreprocessParams& pp,
                                        int orig_w, int orig_h) const;

    static std::vector<YoloBox> nms_per_class(
        std::vector<YoloBox> boxes, float iou_thresh);

    void release();

    int input_w() const { return input_w_; }
    int input_h() const { return input_h_; }
    int num_anchors() const { return num_anchors_; }
    int max_batch_size() const { return max_batch_size_; }
    cudaStream_t stream() const { return stream_; }

private:
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t stream_ = nullptr;

    int input_w_ = 640, input_h_ = 640;
    float conf_thresh_ = 0.25f;
    float nms_iou_ = 0.45f;

    int num_anchors_ = 8400;
    int output_dim_ = 6;  // [x1, y1, x2, y2, conf, cls_id]

    // ── Batch inference state ──
    int max_batch_size_ = 1;
    float* input_gpu_ = nullptr;    // [max_batch, 3, H, W]
    float* output_gpu_ = nullptr;   // [max_batch, N_anchors, out_dim]
    size_t input_bytes_ = 0;
    size_t output_bytes_ = 0;
    int input_per_image_ = 0;       // 3 * W * H
    int output_per_image_ = 0;      // N_anchors * out_dim
    std::string input_name_;
    std::string output_name_;
};

}  // namespace sahi
