/// trt_introspect.cpp — Step 2A: TRT Engine 内省 & 原始输出验证
///
/// 加载 TensorRT engine 文件，打印所有 tensor 的 name/dims/dtype，
/// 跑一次推理并将原始 output tensor 保存为 .bin 文件。
/// 输出文件供 Python 端对比 TRT vs ONNX Runtime 输出是否一致。
///
/// 编译 (Jetson Orin / DeepStream 7.1):
///   g++ -std=c++17 -O2 -o trt_introspect src/trt_introspect.cpp \
///       -I/usr/local/cuda/include -Iinclude \
///       -L/usr/local/cuda/lib64 -lnvinfer -lcudart \
///       $(pkg-config --cflags --libs opencv4)
///
/// 运行:
///   ./trt_introspect /path/to/yolov8n.engine test.jpg [batch_size]

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>

// ===================================================================
// TRT Logger
// ===================================================================
class TrtLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO)
            std::cout << "[TRT] " << msg << std::endl;
    }
};

// ===================================================================
// Preprocess: mirror ultralytics letterbox + normalize
// ===================================================================
static cv::Mat preprocess(const cv::Mat& bgr, int target_w, int target_h,
                          std::vector<float>& blob) {
    float r = std::min(static_cast<float>(target_w) / bgr.cols,
                       static_cast<float>(target_h) / bgr.rows);
    int new_w = static_cast<int>(bgr.cols * r);
    int new_h = static_cast<int>(bgr.rows * r);

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    int pad_w = (target_w - new_w) / 2;
    int pad_h = (target_h - new_h) / 2;
    cv::Mat letterbox(target_h, target_w, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(letterbox(cv::Rect(pad_w, pad_h, new_w, new_h)));

    cv::Mat rgb;
    cv::cvtColor(letterbox, rgb, cv::COLOR_BGR2RGB);
    cv::Mat rgb_float;
    rgb.convertTo(rgb_float, CV_32FC3, 1.0 / 255.0);

    blob.resize(3 * target_w * target_h);
    std::vector<cv::Mat> channels(3);
    cv::split(rgb_float, channels);
    for (int c = 0; c < 3; c++) {
        std::memcpy(blob.data() + c * target_w * target_h,
                    channels[c].data, target_w * target_h * sizeof(float));
    }
    return letterbox;
}

// ===================================================================
// main
// ===================================================================
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <engine_path> <image_path> [batch_override]"
                  << std::endl;
        return 1;
    }

    std::string engine_path = argv[1];
    std::string image_path  = argv[2];
    int batch_override = (argc >= 4) ? std::stoi(argv[3]) : -1;

    TrtLogger logger;

    // ── 1. 加载 Engine ──
    std::ifstream ef(engine_path, std::ios::binary);
    if (!ef) {
        std::cerr << "ERROR: Cannot open engine file: " << engine_path << std::endl;
        return 1;
    }
    ef.seekg(0, std::ios::end);
    size_t engine_size = ef.tellg();
    ef.seekg(0, std::ios::beg);
    std::vector<char> engine_data(engine_size);
    ef.read(engine_data.data(), engine_size);
    ef.close();
    std::cout << "Loaded engine: " << engine_size << " bytes" << std::endl;

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime) {
        std::cerr << "ERROR: Failed to create TRT runtime" << std::endl;
        return 1;
    }

    nvinfer1::ICudaEngine* engine =
        runtime->deserializeCudaEngine(engine_data.data(), engine_size);
    if (!engine) {
        std::cerr << "ERROR: Failed to deserialize engine" << std::endl;
        delete runtime;
        return 1;
    }
    std::cout << "Engine deserialized OK\n" << std::endl;

    // ── 2. 打印 Tensor 信息 ──
    int n_io = engine->getNbIOTensors();
    std::ofstream info("trt_tensor_info.txt");
    info << "Engine: " << engine_path << "\n";
    info << "Total I/O tensors: " << n_io << "\n\n";

    int input_idx = -1;
    std::vector<int> output_indices;

    for (int i = 0; i < n_io; i++) {
        const char* name = engine->getIOTensorName(i);
        auto dims = engine->getTensorShape(name);
        auto dtype = engine->getTensorDataType(name);
        auto mode = engine->getTensorIOMode(name);

        info << "Tensor [" << i << "]: " << name << "\n";
        info << "  IOMode:  "
             << (mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT")
             << "\n";
        info << "  Dims:    [";
        int64_t nelem = 1;
        for (int d = 0; d < dims.nbDims; d++) {
            if (d > 0) info << ", ";
            info << dims.d[d];
            nelem *= dims.d[d];
        }
        info << "]\n";
        info << "  DType:   "
             << (dtype == nvinfer1::DataType::kFLOAT ? "FLOAT"
                    : dtype == nvinfer1::DataType::kHALF  ? "HALF"
                                                          : "INT32")
             << "\n";
        info << "  #Elements: " << nelem << "\n\n";

        std::cout << "[" << i << "] " << name
                  << "  dims=[";
        for (int d = 0; d < dims.nbDims; d++) {
            if (d > 0) std::cout << ",";
            std::cout << dims.d[d];
        }
        std::cout << "]  " << (mode == nvinfer1::TensorIOMode::kINPUT ? "IN" : "OUT")
                  << "  nelem=" << nelem << std::endl;

        if (mode == nvinfer1::TensorIOMode::kINPUT)
            input_idx = i;
        else
            output_indices.push_back(i);
    }
    info.close();
    std::cout << "\nTensor info saved to trt_tensor_info.txt\n" << std::endl;

    if (input_idx < 0) {
        std::cerr << "ERROR: No input tensor found" << std::endl;
        delete engine;
        delete runtime;
        return 1;
    }

    // ── 3. 加载图像并预处理 ──
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "ERROR: Cannot read image: " << image_path << std::endl;
        delete engine;
        delete runtime;
        return 1;
    }
    std::cout << "Image: " << img.cols << "x" << img.rows << std::endl;

    const char* in_name = engine->getIOTensorName(input_idx);
    auto in_dims = engine->getTensorShape(in_name);
    int B = in_dims.d[0];
    int C = in_dims.d[1];
    int H = in_dims.d[2];
    int W = in_dims.d[3];

    if (batch_override > 0 && batch_override <= B) B = batch_override;

    std::cout << "Input shape: [" << B << ", " << C << ", " << H << ", " << W << "]"
              << std::endl;

    std::vector<float> blob;
    cv::Mat letterbox = preprocess(img, W, H, blob);
    cv::imwrite("trt_letterbox.png", letterbox);
    std::cout << "Letterbox preview saved to trt_letterbox.png" << std::endl;

    // 复制到 batch 维度 (同一张图填充整个 batch)
    std::vector<float> batched_blob(B * C * H * W);
    for (int b = 0; b < B; b++) {
        std::memcpy(batched_blob.data() + b * C * H * W,
                    blob.data(), C * H * W * sizeof(float));
    }
    std::cout << "Batched input: " << B << " copies of the same image" << std::endl;

    // ── 4. 分配 GPU Buffer, Enqueue 推理 ──
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    nvinfer1::IExecutionContext* ctx = engine->createExecutionContext();
    if (!ctx) {
        std::cerr << "ERROR: Failed to create execution context" << std::endl;
        delete engine;
        delete runtime;
        return 1;
    }

    // 分配每个 tensor 的 GPU buffer
    std::vector<void*> buffers(n_io, nullptr);
    std::vector<int64_t> buffer_sizes(n_io, 0);

    for (int i = 0; i < n_io; i++) {
        const char* name = engine->getIOTensorName(i);
        auto dims = engine->getTensorShape(name);
        auto dtype = engine->getTensorDataType(name);

        int64_t nelem = 1;
        for (int d = 0; d < dims.nbDims; d++) nelem *= dims.d[d];

        size_t elem_size = (dtype == nvinfer1::DataType::kHALF) ? 2 : 4;
        cudaMalloc(&buffers[i], nelem * elem_size);
        buffer_sizes[i] = nelem;

        ctx->setTensorAddress(name, buffers[i]);
    }

    // 拷贝输入到 GPU
    cudaMemcpyAsync(buffers[input_idx], batched_blob.data(),
                    B * C * H * W * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    // Enqueue
    bool success = ctx->enqueueV3(stream);
    cudaStreamSynchronize(stream);

    if (!success) {
        std::cerr << "ERROR: enqueueV3 failed" << std::endl;
    } else {
        std::cout << "Inference enqueued OK" << std::endl;
    }

    // ── 5. 保存每个 Output Tensor 到 .bin ──
    std::vector<std::string> saved_outputs;
    for (int idx : output_indices) {
        const char* name = engine->getIOTensorName(idx);
        auto dims = engine->getTensorShape(name);
        auto dtype = engine->getTensorDataType(name);

        int64_t nelem = 1;
        for (int d = 0; d < dims.nbDims; d++) nelem *= dims.d[d];

        std::string fname = std::string("raw_trt_") + name + ".bin";

        if (dtype == nvinfer1::DataType::kFLOAT) {
            std::vector<float> cpu_buf(nelem);
            cudaMemcpy(cpu_buf.data(), buffers[idx],
                       nelem * sizeof(float), cudaMemcpyDeviceToHost);

            std::ofstream out(fname, std::ios::binary);
            out.write(reinterpret_cast<const char*>(cpu_buf.data()),
                      nelem * sizeof(float));
            out.close();
        } else if (dtype == nvinfer1::DataType::kHALF) {
            std::vector<float> cpu_buf(nelem * 2);
            cudaMemcpy(cpu_buf.data(), buffers[idx],
                       nelem * sizeof(uint16_t), cudaMemcpyDeviceToHost);

            std::ofstream out(fname, std::ios::binary);
            out.write(reinterpret_cast<const char*>(cpu_buf.data()),
                      nelem * sizeof(uint16_t));
            out.close();
        }

        saved_outputs.push_back(fname);
        std::cout << "Saved " << fname << " (" << nelem << " elements)" << std::endl;
    }

    // 同时保存预处理后的 blob (供步骤 2B 交叉验证)
    {
        std::ofstream out("cpp_blob.bin", std::ios::binary);
        out.write(reinterpret_cast<const char*>(blob.data()),
                  blob.size() * sizeof(float));
        out.close();
        std::cout << "Saved cpp_blob.bin (preprocessed input for step 2B)" << std::endl;
    }

    // ── 清理 ──
    for (int i = 0; i < n_io; i++) cudaFree(buffers[i]);
    cudaStreamDestroy(stream);
    delete ctx;
    delete engine;
    delete runtime;

    std::cout << "\n=== Done ===" << std::endl;
    std::cout << "Output files:" << std::endl;
    std::cout << "  trt_tensor_info.txt   — tensor metadata" << std::endl;
    std::cout << "  trt_letterbox.png     — preprocessed input preview" << std::endl;
    std::cout << "  cpp_blob.bin          — preprocessed float blob" << std::endl;
    for (const auto& fname : saved_outputs) {
        std::cout << "  " << fname << " — raw output data" << std::endl;
    }

    return 0;
}
