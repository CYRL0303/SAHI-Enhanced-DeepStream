#include "sed/deepstream_sahi.hpp"

#include <cmath>
#include <csignal>
#include <fstream>
#include <unistd.h>

namespace sed {

// ===================================================================
// Construction / Destruction
// ===================================================================

DeepStreamSahi::DeepStreamSahi(const DsSahiConfig& config)
    : cfg_(config) {
    DS_LOG("Initializing DeepStream + SAHI framework (C++ pipeline)");

    // Validate required paths
    if (cfg_.nvinfer_config_path.empty()) {
        DS_WARN("nvinfer_config_path is empty — using default");
    }
    if (cfg_.sahi_engine_path.empty() && cfg_.enable_sahi) {
        DS_WARN("SAHI enabled but sahi_engine_path is empty");
    }

    // Initialize GStreamer
    gst_init(nullptr, nullptr);

    // ── SAHI initialisation ──
    if (cfg_.enable_sahi) {
        if (!cfg_.class_labels_path.empty() &&
            !load_class_names(cfg_.class_labels_path)) {
            DS_WARN("SAHI: failed to load class labels, disabling");
            cfg_.enable_sahi = false;
        } else {
            trt_infer_ = std::make_unique<sahi::TrtYoloInfer>();
            if (!trt_infer_->init(cfg_.sahi_engine_path, 640, 640,
                                  cfg_.slice_batch_size,
                                  cfg_.sahi_conf_thresh,
                                  cfg_.sahi_nms_iou)) {
                DS_ERROR("SAHI: TRT engine init failed, falling back to "
                         "full-image-only mode");
                cfg_.enable_sahi = false;
                trt_infer_.reset();
            } else {
                sahi_running_ = true;
                sahi_thread_ = std::thread(&DeepStreamSahi::sahi_loop, this);
                DS_LOG("SAHI enabled  engine=%s  batch=%d  conf=%.2f  "
                       "nms=%.2f  merge=%.2f",
                       cfg_.sahi_engine_path.c_str(),
                       cfg_.slice_batch_size,
                       cfg_.sahi_conf_thresh,
                       cfg_.sahi_nms_iou,
                       cfg_.sahi_merge_iou);
            }
        }
    }

    if (!create_pipeline()) {
        DS_ERROR("Pipeline creation failed");
    }
}

DeepStreamSahi::~DeepStreamSahi() {
    stop();
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    }
}

// ===================================================================
// start / stop
// ===================================================================

bool DeepStreamSahi::start() {
    if (!pipeline_) {
        DS_ERROR("No pipeline to start");
        return false;
    }

    running_ = true;
    gst_element_set_state(pipeline_, GST_STATE_PLAYING);

    main_loop_ = g_main_loop_new(nullptr, FALSE);
    DS_LOG("Pipeline started — entering main loop");
    g_main_loop_run(main_loop_);

    running_ = false;
    gst_element_set_state(pipeline_, GST_STATE_NULL);
    if (main_loop_) {
        g_main_loop_unref(main_loop_);
        main_loop_ = nullptr;
    }
    return true;
}

void DeepStreamSahi::stop() {
    running_ = false;

    if (main_loop_) {
        g_main_loop_quit(main_loop_);
    }

    // Shutdown SAHI
    sahi_running_ = false;
    frame_cv_.notify_all();
    if (sahi_thread_.joinable()) {
        sahi_thread_.join();
    }
    if (trt_infer_) {
        trt_infer_->release();
    }
}

// ===================================================================
// Pipeline creation
// ===================================================================

bool DeepStreamSahi::create_pipeline() {
    std::string W  = std::to_string(cfg_.stream_width);
    std::string H  = std::to_string(cfg_.stream_height);
    std::string sW = std::to_string(cfg_.sahi_frame_width);
    std::string sH = std::to_string(cfg_.sahi_frame_height);

    std::string pipeline_str =
        // Camera 0
        "rtspsrc name=src0 location=" + cfg_.rtsp_url1 + " latency=0 ! "
        "rtph264depay ! h264parse ! nvv4l2decoder ! "
        "tee name=t0 "
        "t0. ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12,"
               "width=" + W + ",height=" + H + " ! "
        "queue ! muxer.sink_0 "
        "t0. ! queue ! nvvidconv ! "
        "video/x-raw,format=BGRx,width=" + sW + ",height=" + sH + " ! "
        "appsink name=sahi_sink0 "
        // Camera 1
        "rtspsrc name=src1 location=" + cfg_.rtsp_url2 + " latency=0 ! "
        "rtph264depay ! h264parse ! nvv4l2decoder ! "
        "tee name=t1 "
        "t1. ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12,"
               "width=" + W + ",height=" + H + " ! "
        "queue ! muxer.sink_1 "
        "t1. ! queue ! nvvidconv ! "
        "video/x-raw,format=BGRx,width=" + sW + ",height=" + sH + " ! "
        "appsink name=sahi_sink1 "
        // nvstreammux + nvinfer + fakesink
        "nvstreammux name=muxer batch-size=2 width=" + W +
        " height=" + H + " ! "
        "nvinfer config-file-path=" + cfg_.nvinfer_config_path +
        " batch-size=2 ! "
        "nvdsosd ! fakesink name=fakesink";

    DS_LOG("Pipeline:\n%s", pipeline_str.c_str());

    GError *error = nullptr;
    pipeline_ = gst_parse_launch(pipeline_str.c_str(), &error);
    if (error) {
        DS_ERROR("Failed to parse pipeline: %s", error->message);
        g_error_free(error);
        return false;
    }

    // Pad probe on fakesink
    GstElement *fakesink = gst_bin_get_by_name(GST_BIN(pipeline_), "fakesink");
    if (!fakesink) {
        DS_ERROR("fakesink not found in pipeline");
        return false;
    }
    GstPad *sinkpad = gst_element_get_static_pad(fakesink, "sink");
    if (!sinkpad) {
        DS_ERROR("Cannot get fakesink sink pad");
        gst_object_unref(fakesink);
        return false;
    }
    gst_pad_add_probe(sinkpad, GST_PAD_PROBE_TYPE_BUFFER,
                      on_buffer_probe, this, nullptr);
    gst_object_unref(sinkpad);
    gst_object_unref(fakesink);

    // Connect SAHI appsinks
    sahi_sink0_ = gst_bin_get_by_name(GST_BIN(pipeline_), "sahi_sink0");
    if (sahi_sink0_) {
        g_object_set(sahi_sink0_, "emit-signals", TRUE, "sync", FALSE, nullptr);
        g_signal_connect(sahi_sink0_, "new-sample",
                         G_CALLBACK(on_appsink_new_sample), this);
        DS_LOG("SAHI appsink0 callback registered");
    }

    sahi_sink1_ = gst_bin_get_by_name(GST_BIN(pipeline_), "sahi_sink1");
    if (sahi_sink1_) {
        g_object_set(sahi_sink1_, "emit-signals", TRUE, "sync", FALSE, nullptr);
        g_signal_connect(sahi_sink1_, "new-sample",
                         G_CALLBACK(on_appsink_new_sample), this);
        DS_LOG("SAHI appsink1 callback registered");
    }

    return true;
}

// ===================================================================
// Pad probe callback
// ===================================================================

GstPadProbeReturn DeepStreamSahi::on_buffer_probe(
    GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
    auto *self = static_cast<DeepStreamSahi*>(user_data);
    GstBuffer *buf = GST_PAD_PROBE_INFO_BUFFER(info);
    self->process_detections(buf);
    return GST_PAD_PROBE_OK;
}

// ===================================================================
// Detection processing (the merge point)
// ===================================================================

void DeepStreamSahi::process_detections(GstBuffer *buf) {
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) {
        DS_WARN("No batch_meta on buffer");
        return;
    }

    int frame_index = 0;
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list;
         l_frame; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;

        FrameResult result;
        result.camera_id = frame_index;
        result.frame_num = frame_meta->frame_num;

        // ── 1. Collect DeepStream full-image detections ──
        std::vector<sahi::YoloBox> full_dets;
        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list;
             l_obj; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
            full_dets.push_back({
                (float)obj_meta->rect_params.left,
                (float)obj_meta->rect_params.top,
                (float)(obj_meta->rect_params.left +
                        obj_meta->rect_params.width),
                (float)(obj_meta->rect_params.top +
                        obj_meta->rect_params.height),
                obj_meta->confidence,
                obj_meta->class_id});
        }

        // ── 2. Read SAHI slice detections ──
        std::vector<sahi::YoloBox> slice_dets;
        if (cfg_.enable_sahi) {
            long age = 0;
            {
                std::lock_guard<std::mutex> lock(sahi_result_mutex_);
                auto &sr = sahi_results_[frame_index];
                age = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - sr.ts).count();
                if (age < cfg_.sahi_result_max_age_ms) {
                    slice_dets = sr.boxes;
                } else if (!sr.boxes.empty()) {
                    DS_LOG("[SAHI-STALE] cam=%d age=%ldms skip",
                           frame_index, age);
                }
            }
        }

        // ── 3. Merge or use full-image only ──
        std::vector<sahi::YoloBox> merged;
        if (!slice_dets.empty()) {
            auto t0 = std::chrono::steady_clock::now();
            merged = sahi::sahi_nms_merge(full_dets, slice_dets,
                                          cfg_.sahi_merge_iou);
            auto t1 = std::chrono::steady_clock::now();
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                t1 - t0).count();
            DS_LOG("[SAHI-MERGE] cam=%d frame=%d full=%zu slice=%zu "
                   "merged=%zu nms_us=%ld",
                   frame_index, frame_meta->frame_num,
                   full_dets.size(), slice_dets.size(), merged.size(), us);
        } else {
            // Use full-image dets as-is (converted to YoloBox and back for
            // consistency with the SAHI merge path)
            for (auto &box : full_dets) {
                merged.push_back(box);
            }
        }

        // ── 4. Build result ──
        for (auto &box : merged) {
            DetectedObject obj;
            obj.x1 = box.x1;
            obj.y1 = box.y1;
            obj.x2 = box.x2;
            obj.y2 = box.y2;
            obj.confidence = box.confidence;
            obj.class_id   = box.class_id;
            obj.class_name = (box.class_id >= 0 &&
                              box.class_id < (int)class_names_.size())
                                 ? class_names_[box.class_id]
                                 : "unknown";
            result.objects.push_back(obj);
        }

        // ── 5. Fire callback ──
        if (callback_ && !result.objects.empty()) {
            callback_(result);
        }

        // Also print summary
        if (!result.objects.empty()) {
            DS_LOG("cam=%d frame=%d dets=%zu",
                   result.camera_id, result.frame_num,
                   result.objects.size());
            for (size_t i = 0; i < result.objects.size(); i++) {
                auto &o = result.objects[i];
                DS_LOG("  det[%zu] cls=%d(%s) conf=%.3f "
                       "bbox=[%.0f,%.0f,%.0f,%.0f]",
                       i, o.class_id, o.class_name.c_str(),
                       o.confidence, o.x1, o.y1, o.x2, o.y2);
            }
        }

        frame_index++;
    }
}

// ===================================================================
// Appsink callback  (SAHI frame capture)
// ===================================================================

GstFlowReturn DeepStreamSahi::on_appsink_new_sample(
    GstElement *sink, gpointer user_data) {
    auto *self = static_cast<DeepStreamSahi*>(user_data);
    GstSample *sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
    if (!sample) return GST_FLOW_OK;

    GstBuffer *buf = gst_sample_get_buffer(sample);
    GstCaps  *caps = gst_sample_get_caps(sample);
    GstStructure *s = gst_caps_get_structure(caps, 0);
    int w = 0, h = 0;
    gst_structure_get_int(s, "width", &w);
    gst_structure_get_int(s, "height", &h);

    GstMapInfo map;
    if (gst_buffer_map(buf, &map, GST_MAP_READ)) {
        cv::Mat bgrx(h, w, CV_8UC4, map.data);
        cv::Mat bgr;
        cv::cvtColor(bgrx, bgr, cv::COLOR_BGRA2BGR);

        int cam = (sink == self->sahi_sink0_) ? 0 : 1;

        if (self->cfg_.enable_sahi) {
            std::lock_guard<std::mutex> lock(self->frame_queue_mutex_);
            auto &q = self->frame_queues_[cam];
            if ((int)q.size() >= self->kFrameQueueDepth) {
                q.pop_front();
            }
            uint64_t seq = self->frame_seqs_[cam]++;
            auto now = std::chrono::steady_clock::now();
            q.push_back({bgr.clone(), seq, now});
            self->frame_cv_.notify_one();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()).count();
            DS_LOG("[SAHI-QUEUE] cam=%d seq=%lu ts=%ld", cam, seq, ms);
        }

        gst_buffer_unmap(buf, &map);
    }
    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

// ===================================================================
// SAHI thread loop
// ===================================================================

void DeepStreamSahi::sahi_loop() {
    DS_LOG("SAHI thread started");

    while (sahi_running_) {
        SFrame frames[kMaxCameras];
        bool has_frame[kMaxCameras] = {false, false};

        {
            std::unique_lock<std::mutex> lock(frame_queue_mutex_);
            frame_cv_.wait_for(lock, std::chrono::milliseconds(100), [this]() {
                return !sahi_running_ ||
                       !frame_queues_[0].empty() ||
                       !frame_queues_[1].empty();
            });

            if (!sahi_running_) break;

            for (int cam = 0; cam < kMaxCameras; cam++) {
                if (!frame_queues_[cam].empty()) {
                    frames[cam] = std::move(frame_queues_[cam].front());
                    frame_queues_[cam].pop_front();
                    has_frame[cam] = true;
                }
            }
        }

        for (int cam = 0; cam < kMaxCameras; cam++) {
            if (!has_frame[cam]) continue;

            auto t0 = std::chrono::steady_clock::now();
            DS_LOG("[SAHI-PROC] cam=%d seq=%lu start",
                   cam, frames[cam].seq);

            auto dets = process_sahi_frame(cam, frames[cam].bgr);

            auto t1 = std::chrono::steady_clock::now();
            auto proc_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                t1 - t0).count();

            std::lock_guard<std::mutex> lock(sahi_result_mutex_);
            sahi_results_[cam] = {std::move(dets), frames[cam].seq,
                                  frames[cam].ts};

            DS_LOG("[SAHI-PROC] cam=%d seq=%lu done dets=%zu ms=%ld",
                   cam, frames[cam].seq,
                   sahi_results_[cam].boxes.size(), proc_ms);
        }
    }

    DS_LOG("SAHI thread exiting");
}

// ===================================================================
// Per-frame SAHI processing
// ===================================================================

std::vector<sahi::YoloBox> DeepStreamSahi::process_sahi_frame(
    int camera_id, const cv::Mat& bgr) {
    if (!trt_infer_) return {};

    // (Re)create slicer if resolution changed
    if (!slicers_[camera_id] ||
        slicers_[camera_id]->image_w() != bgr.cols ||
        slicers_[camera_id]->image_h() != bgr.rows) {
        slicers_[camera_id] = std::make_unique<sahi::SahiSlicer>(
            cfg_.sahi_slice_h, cfg_.sahi_slice_w,
            cfg_.sahi_overlap_h, cfg_.sahi_overlap_w,
            bgr.rows, bgr.cols);
        DS_LOG("SAHI cam%d slicer: %dx%d -> %d slices",
               camera_id, bgr.cols, bgr.rows,
               slicers_[camera_id]->num_slices());
    }

    auto slices = slicers_[camera_id]->compute_slices();

    std::vector<cv::Mat>       batch_imgs;
    std::vector<sahi::Slice>   batch_slices;
    std::vector<sahi::YoloBox> all_dets;

    for (size_t i = 0; i < slices.size(); i++) {
        cv::Rect roi(slices[i].x_min, slices[i].y_min,
                     slices[i].width(), slices[i].height());
        batch_imgs.push_back(bgr(roi).clone());
        batch_slices.push_back(slices[i]);

        if ((int)batch_imgs.size() >= cfg_.slice_batch_size ||
            i == slices.size() - 1) {
            auto batch_results = trt_infer_->infer_batch(batch_imgs);
            for (size_t b = 0; b < batch_results.size(); b++) {
                for (auto &det : batch_results[b]) {
                    det.x1 += batch_slices[b].x_min;
                    det.y1 += batch_slices[b].y_min;
                    det.x2 += batch_slices[b].x_min;
                    det.y2 += batch_slices[b].y_min;
                    all_dets.push_back(det);
                }
            }
            batch_imgs.clear();
            batch_slices.clear();
        }
    }

    return all_dets;
}

// ===================================================================
// Class name loading
// ===================================================================

bool DeepStreamSahi::load_class_names(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        DS_ERROR("SAHI: cannot open class labels: %s", path.c_str());
        return false;
    }
    class_names_.clear();
    std::string line;
    while (std::getline(file, line)) {
        while (!line.empty() &&
               (line.back() == '\r' || line.back() == ' '))
            line.pop_back();
        if (!line.empty())
            class_names_.push_back(line);
    }
    DS_LOG("SAHI: loaded %zu class names", class_names_.size());
    return !class_names_.empty();
}

} // namespace sed
