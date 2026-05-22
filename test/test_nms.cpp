/// test_nms.cpp — Step 3 NMS merge verification
///
/// Two modes:
///   [no args]  Synthetic unit tests (8 cases)
///   [with args] JSON merge: <full.json> <slices.json> <output.json> [iou=0.5]
///
/// Build: links trt_yolo_infer_obj (requires CUDA/TensorRT headers)

#include "sed/sahi_nms.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

// ===================================================================
// Minimal JSON helpers (avoid dependency on a JSON library)
// ===================================================================
namespace {

std::string read_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "ERROR: Cannot open " << path << std::endl;
        return {};
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void write_json(const std::string& path,
                const std::vector<sahi::YoloBox>& boxes) {
    std::ofstream out(path);
    out << "{\n  \"boxes\": [\n";
    for (size_t i = 0; i < boxes.size(); i++) {
        auto& b = boxes[i];
        out << "    {";
        out << "\"x1\":" << b.x1
            << ",\"y1\":" << b.y1
            << ",\"x2\":" << b.x2
            << ",\"y2\":" << b.y2
            << ",\"confidence\":" << b.confidence
            << ",\"class_id\":" << b.class_id;
        out << "}";
        if (i + 1 < boxes.size()) out << ",";
        out << "\n";
    }
    out << "  ]\n}\n";
}

// Parse one float value after "key": from JSON
float parse_float_after(const std::string& json, const std::string& key,
                        size_t& pos) {
    pos = json.find(key, pos);
    if (pos == std::string::npos) return 0.0f;
    pos = json.find(':', pos) + 1;
    return std::stof(json.substr(pos));
}

int parse_int_after(const std::string& json, const std::string& key,
                    size_t& pos) {
    pos = json.find(key, pos);
    if (pos == std::string::npos) return 0;
    pos = json.find(':', pos) + 1;
    return std::stoi(json.substr(pos));
}

std::vector<sahi::YoloBox> parse_boxes_json(const std::string& json) {
    std::vector<sahi::YoloBox> boxes;
    size_t pos = 0;
    while (true) {
        pos = json.find("\"x1\"", pos);
        if (pos == std::string::npos) break;
        sahi::YoloBox b;
        b.x1 = parse_float_after(json, "\"x1\"", pos);
        b.y1 = parse_float_after(json, "\"y1\"", pos);
        b.x2 = parse_float_after(json, "\"x2\"", pos);
        b.y2 = parse_float_after(json, "\"y2\"", pos);
        b.confidence = parse_float_after(json, "\"confidence\"", pos);
        b.class_id = parse_int_after(json, "\"class_id\"", pos);
        boxes.push_back(b);
    }
    return boxes;
}

int json_merge_mode(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <full.json> <slices.json> <output.json> [iou=0.5]"
                  << std::endl;
        return 1;
    }
    std::string full_json_path = argv[1];
    std::string slices_json_path = argv[2];
    std::string output_json_path = argv[3];
    float iou = (argc >= 5) ? std::stof(argv[4]) : 0.5f;

    auto full_dets = parse_boxes_json(read_file(full_json_path));
    auto slice_dets = parse_boxes_json(read_file(slices_json_path));

    std::cout << "Full: " << full_dets.size()
              << " boxes, Slices: " << slice_dets.size() << " boxes"
              << std::endl;

    auto merged = sahi::sahi_nms_merge(full_dets, slice_dets, iou);

    std::cout << "Merged: " << merged.size() << " boxes (iou=" << iou << ")"
              << std::endl;

    write_json(output_json_path, merged);
    std::cout << "Saved " << output_json_path << std::endl;
    return 0;
}

}  // namespace

// ===================================================================
// Synthetic unit tests
// ===================================================================
namespace {

int run_tests() {
    int passed = 0, failed = 0;

    auto check = [&](const char* name, bool cond) {
        if (cond) {
            std::cout << "  PASS: " << name << std::endl;
            passed++;
        } else {
            std::cerr << "  FAIL: " << name << std::endl;
            failed++;
        }
    };

    // ── Test 1: Both empty ──
    {
        std::cout << "\n=== Test 1: Both empty ===" << std::endl;
        auto r = sahi::sahi_nms_merge({}, {}, 0.5f);
        check("returns empty", r.empty());
    }

    // ── Test 2: Only full dets ──
    {
        std::cout << "\n=== Test 2: Only full dets ===" << std::endl;
        std::vector<sahi::YoloBox> full = {
            {100, 100, 200, 200, 0.9f, 0},
        };
        auto r = sahi::sahi_nms_merge(full, {}, 0.5f);
        check("count=1", r.size() == 1);
        check("same box", r[0].x1 == 100 && r[0].confidence == 0.9f);
    }

    // ── Test 3: Only slice dets ──
    {
        std::cout << "\n=== Test 3: Only slice dets ===" << std::endl;
        std::vector<sahi::YoloBox> slices = {
            {500, 100, 520, 120, 0.6f, 0},
        };
        auto r = sahi::sahi_nms_merge({}, slices, 0.5f);
        check("count=1", r.size() == 1);
        check("same box", r[0].x1 == 500 && r[0].confidence == 0.6f);
    }

    // ── Test 4: Overlapping same class — keep highest confidence ──
    {
        std::cout << "\n=== Test 4: Overlapping same class ===" << std::endl;
        std::vector<sahi::YoloBox> full = {
            {100, 200, 300, 400, 0.9f, 0},   // 全图检测到的大船
        };
        std::vector<sahi::YoloBox> slices = {
            {110, 210, 290, 390, 0.7f, 0},   // 切片检测到同一艘 (应被抑制)
        };
        auto r = sahi::sahi_nms_merge(full, slices, 0.5f);
        check("count=1 (duplicate suppressed)", r.size() == 1);
        check("kept higher confidence", r[0].confidence == 0.9f);
        check("kept correct box", r[0].x1 == 100 && r[0].class_id == 0);
    }

    // ── Test 5: Non-overlapping same class — keep both ──
    {
        std::cout << "\n=== Test 5: Non-overlapping same class ===" << std::endl;
        std::vector<sahi::YoloBox> full = {
            {100, 200, 300, 400, 0.8f, 0},   // 近处大船
        };
        std::vector<sahi::YoloBox> slices = {
            {500, 100, 520, 120, 0.6f, 0},   // 远处小船, 不重叠
        };
        auto r = sahi::sahi_nms_merge(full, slices, 0.5f);
        check("count=2 (both kept)", r.size() == 2);
    }

    // ── Test 6: Overlapping different classes — keep both ──
    {
        std::cout << "\n=== Test 6: Overlapping different classes ===" << std::endl;
        std::vector<sahi::YoloBox> full = {
            {100, 200, 300, 400, 0.9f, 0},   // 船 (class 0)
        };
        std::vector<sahi::YoloBox> slices = {
            {110, 210, 290, 390, 0.7f, 1},   // 人 (class 1), 和船重叠
        };
        auto r = sahi::sahi_nms_merge(full, slices, 0.5f);
        check("count=2 (different classes)", r.size() == 2);
        check("boat kept", r[0].class_id == 0 || r[1].class_id == 0);
        check("person kept", r[0].class_id == 1 || r[1].class_id == 1);
    }

    // ── Test 7: Multi-class mixed scenario (from the implementation plan) ──
    {
        std::cout << "\n=== Test 7: Multi-class mixed ===" << std::endl;
        std::vector<sahi::YoloBox> full = {
            {100, 200, 300, 400, 0.9f, 0},   // 近处大船
            {600, 700, 800, 900, 0.8f, 1},   // 人
        };
        std::vector<sahi::YoloBox> slices = {
            {110, 210, 290, 390, 0.7f, 0},   // 同一艘大船 (应被抑制)
            {500, 100, 520, 120, 0.6f, 0},   // 远处小船 (应保留)
            {610, 710, 790, 890, 0.5f, 1},   // 同一个人 (应被抑制)
        };
        auto r = sahi::sahi_nms_merge(full, slices, 0.5f);
        check("count=3 (2 boat + 1 person)", r.size() == 3);

        // 统计各类数量
        int boat = 0, person = 0;
        for (auto& b : r) {
            if (b.class_id == 0) boat++;
            if (b.class_id == 1) person++;
        }
        check("2 boats", boat == 2);
        check("1 person", person == 1);

        // 远处小船 (x≈500) 必须保留
        bool has_small_boat = false;
        for (auto& b : r) {
            if (b.class_id == 0 && b.x1 > 400) has_small_boat = true;
        }
        check("distant small boat kept", has_small_boat);
    }

    // ── Test 8: IoU boundary — exactly at threshold ──
    {
        std::cout << "\n=== Test 8: IoU boundary ===" << std::endl;
        // 100x100 box: (100,100)-(200,200) area=10000
        // To get IoU=0.5, overlap area must be 10000*0.5/1.5 = 3333.3
        // Box shifted by ~42px: (142,100)-(242,200) → overlap=58*100=5800
        //   IoU = 5800/(10000+10000-5800) = 5800/14200 = 0.408
        // Too low. Let me calculate properly.
        // Want IoU exactly at threshold, say 0.50.
        // A=(100,100)-(200,200), area=10000
        // B=(100+dx,100)-(200+dx,200), area=10000
        // overlap_w = max(0, 200-(100+dx)) = max(0, 100-dx)
        // overlap = (100-dx)*100 = 10000-100*dx
        // IoU = (10000-100dx) / (20000-(10000-100dx)) = (10000-100dx)/(10000+100dx)
        // For IoU=0.500: (10000-100dx) = 0.5*(10000+100dx) = 5000+50dx
        // 10000-100dx = 5000+50dx → 5000 = 150dx → dx = 33.333
        // So dx=33 → overlap=6700, IoU=6700/13300=0.5038 → suppressed
        // dx=34 → overlap=6600, IoU=6600/13400=0.4925 → kept
        // dx=50 → overlap=5000, IoU=5000/15000=0.3333 → kept

        // Sub-test A: IoU ≈ 0.503 > 0.50 → suppressed
        {
            std::vector<sahi::YoloBox> full = {
                {100, 100, 200, 200, 0.9f, 0},
            };
            std::vector<sahi::YoloBox> slices = {
                {133, 100, 233, 200, 0.7f, 0},  // IoU ≈ 0.504 > 0.5
            };
            auto r = sahi::sahi_nms_merge(full, slices, 0.5f);
            check("IoU>0.5 suppressed", r.size() == 1);
        }

        // Sub-test B: IoU ≈ 0.493 < 0.50 → kept
        {
            std::vector<sahi::YoloBox> full = {
                {100, 100, 200, 200, 0.9f, 0},
            };
            std::vector<sahi::YoloBox> slices = {
                {134, 100, 234, 200, 0.7f, 0},  // IoU ≈ 0.493 < 0.5
            };
            auto r = sahi::sahi_nms_merge(full, slices, 0.5f);
            check("IoU<0.5 kept", r.size() == 2);
        }
    }

    // ── Test 9: Chain suppression (3 overlapping boxes) ──
    {
        std::cout << "\n=== Test 9: Chain suppression ===" << std::endl;
        std::vector<sahi::YoloBox> full = {
            {100, 100, 200, 200, 0.95f, 0},   // A: 最高
        };
        std::vector<sahi::YoloBox> slices = {
            {110, 110, 210, 210, 0.80f, 0},   // B: 和 A 重叠 → 抑制
            {105, 105, 205, 205, 0.60f, 0},   // C: 和 A 重叠 → 抑制
        };
        auto r = sahi::sahi_nms_merge(full, slices, 0.5f);
        check("count=1 (chain suppressed)", r.size() == 1);
        check("highest kept", r[0].confidence == 0.95f);
    }

    // ── Test 10: Slice det has higher confidence ──
    {
        std::cout << "\n=== Test 10: Slice higher confidence ===" << std::endl;
        std::vector<sahi::YoloBox> full = {
            {100, 200, 300, 400, 0.6f, 0},   // 全图置信度低
        };
        std::vector<sahi::YoloBox> slices = {
            {110, 210, 290, 390, 0.9f, 0},   // 切片置信度更高 (FP16 噪声或特征更清晰)
        };
        auto r = sahi::sahi_nms_merge(full, slices, 0.5f);
        check("count=1", r.size() == 1);
        check("kept higher confidence from slice", r[0].confidence == 0.9f);
    }

    // ── Summary ──
    std::cout << "\n=== Summary: " << passed << "/" << (passed + failed)
              << " PASSED ===" << std::endl;
    return (failed == 0) ? 0 : 1;
}

}  // namespace

// ===================================================================
// main — dispatch
// ===================================================================
int main(int argc, char** argv) {
    if (argc >= 4) {
        return json_merge_mode(argc, argv);
    }
    return run_tests();
}
