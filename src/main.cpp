/// SAHI Enhanced DeepStream (SED) — standalone demo application.

#include "sed/deepstream_sahi.hpp"

#include <csignal>
#include <cstdlib>

namespace {

sed::DeepStreamSahi *g_app = nullptr;

void signal_handler(int) {
    if (g_app) g_app->stop();
}

} // namespace

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    signal(SIGINT,  signal_handler);
    signal(SIGTERM, signal_handler);

    // ── Configure ──
    sed::DsSahiConfig cfg;

    cfg.rtsp_url1 = "rtsp://172.16.1.17:8554/live/stream";
    cfg.rtsp_url2 = "rtsp://172.16.1.17:8554/live/stream";

    cfg.stream_width  = 640;
    cfg.stream_height = 480;

    cfg.sahi_frame_width  = 1920;
    cfg.sahi_frame_height = 1080;

    cfg.nvinfer_config_path =
        "./config/deepstream_config_yolo.txt";

    cfg.enable_sahi       = true;
    cfg.sahi_engine_path  = "./config/yolov8n_batch.engine";
    cfg.class_labels_path = "./config/labels.txt";

    cfg.slice_batch_size = 4;
    cfg.sahi_conf_thresh = 0.25f;
    cfg.sahi_nms_iou     = 0.45f;
    cfg.sahi_merge_iou   = 0.5f;
    cfg.sahi_slice_h     = 640;
    cfg.sahi_slice_w     = 640;
    cfg.sahi_overlap_h   = 0.2f;
    cfg.sahi_overlap_w   = 0.2f;
    cfg.sahi_result_max_age_ms = 500;

    // ── Build ──
    sed::DeepStreamSahi app(cfg);
    g_app = &app;

    // ── Optional: set user callback ──
    app.set_callback([](const sed::FrameResult &r) {
        // Replace this with your own logic — publish to a message bus,
        // write to a database, send over TCP, etc.
        printf("[CALLBACK] cam=%d frame=%d objects=%zu\n",
               r.camera_id, r.frame_num, r.objects.size());
    });

    // ── Run ──
    if (!app.start()) {
        fprintf(stderr, "Failed to start pipeline\n");
        return 1;
    }

    printf("Pipeline exited normally.\n");
    return 0;
}
