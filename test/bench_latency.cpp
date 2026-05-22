/// bench_latency.cpp — SAHI slice inference latency benchmark
/// Measures per-stage timing for batch=6 and batch=8 on Jetson Orin.
///
/// Usage:
///   ./bench_latency <engine_path> <image_path> [num_warmup=10] [num_iter=50]

#include "sed/trt_yolo_infer.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

static double ms_since(Clock::time_point start) {
    auto now = Clock::now();
    return std::chrono::duration<double, std::milli>(now - start).count();
}

struct Stats {
    double min = 0, max = 0, mean = 0, median = 0;
};

static Stats compute_stats(std::vector<double>& samples) {
    if (samples.empty()) return {};
    std::sort(samples.begin(), samples.end());
    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    return { samples.front(), samples.back(), sum / samples.size(), samples[samples.size() / 2] };
}

static void print_stats(const std::string& label, const Stats& s) {
    printf("  %-22s min=%7.2f  med=%7.2f  avg=%7.2f  max=%7.2f ms\n",
           label.c_str(), s.min, s.median, s.mean, s.max);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <engine_path> <image> [warmup=10] [iter=50]" << std::endl;
        return 1;
    }
    std::string engine_path = argv[1];
    std::string image_path  = argv[2];
    int num_warmup = (argc >= 4) ? std::stoi(argv[3]) : 10;
    int num_iter   = (argc >= 5) ? std::stoi(argv[4]) : 50;

    cv::Mat src = cv::imread(image_path);
    if (src.empty()) { std::cerr << "ERROR: Cannot read " << image_path << std::endl; return 1; }
    printf("Image: %dx%d  Warmup: %d  Iter: %d\n\n", src.cols, src.rows, num_warmup, num_iter);

    sahi::TrtYoloInfer infer;
    if (!infer.init(engine_path, 640, 640, 8, 0.25f, 0.45f)) {
        std::cerr << "ERROR: init failed" << std::endl;
        return 1;
    }

    // CUDA events on the inference stream
    cudaEvent_t evt_start, evt_stop;
    cudaEventCreate(&evt_start);
    cudaEventCreate(&evt_stop);

    for (int batch : {1, 6, 8}) {
        printf("══════ Batch=%-2d ══════\n", batch);
        std::vector<cv::Mat> slices(batch);
        for (int i = 0; i < batch; i++) slices[i] = src.clone();

        std::vector<double> t_wall, t_gpu, t_cpu;

        // Warmup
        for (int w = 0; w < num_warmup; w++) infer.infer_batch(slices);
        cudaDeviceSynchronize();

        for (int iter = 0; iter < num_iter; iter++) {
            // Record CUDA event BEFORE any work is queued on the stream
            cudaEventRecord(evt_start, infer.stream());

            auto t0 = Clock::now();
            auto results = infer.infer_batch(slices);
            auto t1 = Clock::now();

            // Record CUDA event AFTER all GPU work on the stream
            cudaEventRecord(evt_stop, infer.stream());
            cudaEventSynchronize(evt_stop);

            float gpu_ms = 0;
            cudaEventElapsedTime(&gpu_ms, evt_start, evt_stop);

            double wall_ms = ms_since(t0);
            // GPU time = preprocess upload + TRT enqueue
            // CPU time within infer_batch = letterbox + decode + NMS + overhead
            // (GPU time is measured on-stream, CPU time = wall - GPU overlap)

            t_wall.push_back(wall_ms);
            t_gpu.push_back(gpu_ms);
        }

        Stats sw = compute_stats(t_wall);
        Stats sg = compute_stats(t_gpu);

        print_stats("Total wall-clock", sw);
        print_stats("GPU (upload+TRT)", sg);

        double fps_med = 1000.0 / sw.median;
        double fps_avg = 1000.0 / sw.mean;
        printf("  →  FPS:  median=%.1f  avg=%.1f  (wall clock)\n\n", fps_med, fps_avg);
    }

    cudaEventDestroy(evt_start);
    cudaEventDestroy(evt_stop);
    infer.release();
    printf("Done.\n");
    return 0;
}
