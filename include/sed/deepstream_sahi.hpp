#pragma once

/// Standalone DeepStream + SAHI integration framework.
/// Zero ROS2 dependencies — pure GStreamer / DeepStream / TensorRT / OpenCV.
///
/// Design mirrors ship_assistant::DeepStreamNode but replaces:
///   - rclcpp::Node        → DsSahiConfig struct
///   - ROS2 publisher      → DetectionCallback (user-supplied)
///   - RCLCPP_* logging    → DS_LOG / DS_WARN / DS_ERROR macros
///   - ROS2 parameters     → DsSahiConfig fields

#include "sed/sahi_slicer.hpp"
#include "sed/trt_yolo_infer.hpp"
#include "sed/sahi_nms.hpp"

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <nvdsmeta.h>
#include <gstnvdsmeta.h>
#include <opencv2/opencv.hpp>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace sed {

// ── Logging (replaces RCLCPP_*) ──
#define DS_LOG(fmt, ...)  printf("[DeepStreamSahi] " fmt "\n", ##__VA_ARGS__)
#define DS_WARN(fmt, ...) fprintf(stderr, "[DeepStreamSahi WARN] " fmt "\n", ##__VA_ARGS__)
#define DS_ERROR(fmt, ...) fprintf(stderr, "[DeepStreamSahi ERROR] " fmt "\n", ##__VA_ARGS__)

// ── Detection output (replaces VesselDetection ROS2 msg) ──
struct DetectedObject {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
    std::string class_name;
    int track_id = -1;
};

struct FrameResult {
    int camera_id;
    int frame_num;
    std::vector<DetectedObject> objects;
};

using DetectionCallback = std::function<void(const FrameResult&)>;

// ── Configuration (replaces ROS2 parameters) ──
struct DsSahiConfig {
    std::string rtsp_url1 = "rtsp://172.16.1.17:8554/live/stream";
    std::string rtsp_url2 = "rtsp://172.16.1.17:8554/live/stream";

    int stream_width  = 640;
    int stream_height = 480;

    int sahi_frame_width  = 1920;
    int sahi_frame_height = 1080;

    std::string nvinfer_config_path;

    bool        enable_sahi         = false;
    std::string sahi_engine_path;
    std::string class_labels_path;

    int   slice_batch_size      = 4;
    float sahi_conf_thresh      = 0.25f;
    float sahi_nms_iou          = 0.45f;
    float sahi_merge_iou        = 0.5f;
    int   sahi_slice_h          = 640;
    int   sahi_slice_w          = 640;
    float sahi_overlap_h        = 0.2f;
    float sahi_overlap_w        = 0.2f;
    int   sahi_result_max_age_ms = 500;
};

// ── Main class ──
class DeepStreamSahi {
public:
    explicit DeepStreamSahi(const DsSahiConfig& config);
    ~DeepStreamSahi();

    DeepStreamSahi(const DeepStreamSahi&) = delete;
    DeepStreamSahi& operator=(const DeepStreamSahi&) = delete;

    void set_callback(DetectionCallback cb) { callback_ = std::move(cb); }

    bool start();          // blocks in GMainLoop until stop() or error
    void stop();
    bool is_running() const { return running_; }

private:
    DsSahiConfig cfg_;
    DetectionCallback callback_;

    // ── GStreamer ──
    GstElement *pipeline_    = nullptr;
    GstElement *tee0_        = nullptr;
    GstElement *tee1_        = nullptr;
    GstElement *sahi_sink0_  = nullptr;
    GstElement *sahi_sink1_  = nullptr;
    GMainLoop   *main_loop_  = nullptr;
    std::thread  pipeline_thread_;
    std::atomic<bool> running_{false};

    // ── SAHI frame queue ──
    struct SFrame {
        cv::Mat bgr;
        uint64_t seq;
        std::chrono::steady_clock::time_point ts;
    };
    struct SResult {
        std::vector<sahi::YoloBox> boxes;
        uint64_t seq;
        std::chrono::steady_clock::time_point ts;
    };

    static constexpr int kMaxCameras     = 2;
    static constexpr int kFrameQueueDepth = 2;

    std::mutex              frame_queue_mutex_;
    std::condition_variable frame_cv_;
    std::deque<SFrame>      frame_queues_[kMaxCameras];
    uint64_t                frame_seqs_[kMaxCameras] = {0, 0};

    std::mutex sahi_result_mutex_;
    SResult    sahi_results_[kMaxCameras];

    // ── SAHI components ──
    std::unique_ptr<sahi::TrtYoloInfer> trt_infer_;
    std::unique_ptr<sahi::SahiSlicer>   slicers_[kMaxCameras];

    std::thread        sahi_thread_;
    std::atomic<bool>  sahi_running_{false};
    void sahi_loop();
    std::vector<sahi::YoloBox> process_sahi_frame(int camera_id, const cv::Mat& bgr);

    std::vector<std::string> class_names_;
    bool load_class_names(const std::string& path);

    // ── Pipeline ──
    bool create_pipeline();

    static GstPadProbeReturn on_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);
    static GstFlowReturn     on_appsink_new_sample(GstElement *sink, gpointer user_data);

    void process_detections(GstBuffer *buf);
};

} // namespace sed
