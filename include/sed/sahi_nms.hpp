#pragma once

#include "sed/trt_yolo_infer.hpp"

#include <vector>

namespace sahi {

// 双源 NMS 合并：全图检测 + 切片检测 → 去重统一结果
// full_dets:  全图推理结果（坐标已是原图空间）
// slice_dets: 切片推理结果（坐标已平移到原图空间）
// iou_threshold: 合并 IoU 阈值，推荐 0.5
std::vector<YoloBox> sahi_nms_merge(
    const std::vector<YoloBox>& full_dets,
    const std::vector<YoloBox>& slice_dets,
    float iou_threshold);

}  // namespace sahi
