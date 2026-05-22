#include "sed/sahi_nms.hpp"

namespace sahi {

std::vector<YoloBox> sahi_nms_merge(
    const std::vector<YoloBox>& full_dets,
    const std::vector<YoloBox>& slice_dets,
    float iou_threshold)
{
    std::vector<YoloBox> merged;
    merged.reserve(full_dets.size() + slice_dets.size());
    merged.insert(merged.end(), full_dets.begin(), full_dets.end());
    merged.insert(merged.end(), slice_dets.begin(), slice_dets.end());
    return TrtYoloInfer::nms_per_class(std::move(merged), iou_threshold);
}

}  // namespace sahi
