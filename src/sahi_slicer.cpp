#include "sed/sahi_slicer.hpp"

#include <algorithm>

namespace sahi {

// ===================================================================
// Public API
// ===================================================================

SahiSlicer::SahiSlicer(int slice_h, int slice_w,
                       float overlap_h_ratio, float overlap_w_ratio,
                       int image_h, int image_w)
    : slice_h_(slice_h)
    , slice_w_(slice_w)
    , overlap_h_pixels_(static_cast<int>(overlap_h_ratio * slice_h))
    , overlap_w_pixels_(static_cast<int>(overlap_w_ratio * slice_w))
    , image_h_(image_h)
    , image_w_(image_w)
{}

SahiSlicer::SahiSlicer(int slice_h, int slice_w,
                       int overlap_h_pixels, int overlap_w_pixels,
                       int image_h, int image_w, bool)
    : slice_h_(slice_h)
    , slice_w_(slice_w)
    , overlap_h_pixels_(overlap_h_pixels)
    , overlap_w_pixels_(overlap_w_pixels)
    , image_h_(image_h)
    , image_w_(image_w)
{}

SahiSlicer SahiSlicer::from_auto(int image_h, int image_w) {
    AutoParams p = auto_params(image_h, image_w);
    return SahiSlicer(p.slice_h, p.slice_w,
                      p.overlap_h_pixels, p.overlap_w_pixels,
                      image_h, image_w, false);
}

std::vector<Slice> SahiSlicer::compute_slices() const {
    if (slices_computed_)
        return cached_slices_;

    std::vector<Slice> slices;
    int y_max = 0, y_min = 0;

    while (y_max < image_h_) {
        int x_max = 0, x_min = 0;
        y_max = y_min + slice_h_;

        while (x_max < image_w_) {
            x_max = x_min + slice_w_;

            if (y_max > image_h_ || x_max > image_w_) {
                x_max = std::min(image_w_, x_max);
                y_max = std::min(image_h_, y_max);
                x_min = std::max(0, x_max - slice_w_);
                y_min = std::max(0, y_max - slice_h_);
                slices.push_back({x_min, y_min, x_max, y_max});
            } else {
                slices.push_back({x_min, y_min, x_max, y_max});
            }

            x_min = x_max - overlap_w_pixels_;
        }
        y_min = y_max - overlap_h_pixels_;
    }

    cached_slices_ = slices;
    slices_computed_ = true;
    return cached_slices_;
}

int SahiSlicer::num_slices() const {
    if (!slices_computed_)
        compute_slices();
    return static_cast<int>(cached_slices_.size());
}

// ===================================================================
// Auto-parameter resolution (ported from sahi/slicing.py)
// ===================================================================

AutoParams SahiSlicer::auto_params(int image_h, int image_w) {
    int resolution = image_h * image_w;
    int factor = calc_resolution_factor(resolution);

    int x_overlap, y_overlap, slice_width, slice_height;

    if (factor <= 18) {
        get_resolution_selector("low", image_h, image_w,
                                x_overlap, y_overlap, slice_width, slice_height);
    } else if (factor < 21) {
        get_resolution_selector("medium", image_h, image_w,
                                x_overlap, y_overlap, slice_width, slice_height);
    } else if (factor < 24) {
        get_resolution_selector("high", image_h, image_w,
                                x_overlap, y_overlap, slice_width, slice_height);
    } else {
        get_resolution_selector("ultra-high", image_h, image_w,
                                x_overlap, y_overlap, slice_width, slice_height);
    }

    return {slice_height, slice_width, y_overlap, x_overlap};
}

// ===================================================================
// Static helpers
// ===================================================================

int SahiSlicer::calc_resolution_factor(int resolution) {
    int expo = 0;
    while ((1 << expo) < resolution)
        expo++;
    return expo - 1;
}

SahiSlicer::Orientation SahiSlicer::calc_orientation(int width, int height) {
    if (width < height) return Orientation::kVertical;
    if (width > height) return Orientation::kHorizontal;
    return Orientation::kSquare;
}

void SahiSlicer::calc_ratio_and_slice(Orientation orientation, int slide, float ratio,
                                       int& slice_row, int& slice_col,
                                       float& overlap_h_ratio, float& overlap_w_ratio) {
    switch (orientation) {
        case Orientation::kVertical:
            slice_row = slide; slice_col = slide * 2;
            break;
        case Orientation::kHorizontal:
            slice_row = slide * 2; slice_col = slide;
            break;
        case Orientation::kSquare:
            slice_row = slide; slice_col = slide;
            break;
    }
    overlap_h_ratio = ratio;
    overlap_w_ratio = ratio;
}

void SahiSlicer::calc_slice_and_overlap_params(const std::string& resolution,
                                                int height, int width,
                                                Orientation orientation,
                                                int& x_overlap, int& y_overlap,
                                                int& slice_width, int& slice_height) {
    int split_row, split_col;
    float overlap_h_ratio, overlap_w_ratio;

    if (resolution == "medium") {
        calc_ratio_and_slice(orientation, 1, 0.8f,
                             split_row, split_col,
                             overlap_h_ratio, overlap_w_ratio);
    } else if (resolution == "high") {
        calc_ratio_and_slice(orientation, 2, 0.4f,
                             split_row, split_col,
                             overlap_h_ratio, overlap_w_ratio);
    } else if (resolution == "ultra-high") {
        calc_ratio_and_slice(orientation, 4, 0.4f,
                             split_row, split_col,
                             overlap_h_ratio, overlap_w_ratio);
    } else {  // "low"
        split_col = 1;
        split_row = 1;
        overlap_w_ratio = 1.0f;
        overlap_h_ratio = 1.0f;
    }

    slice_height = height / split_col;
    slice_width  = width  / split_row;

    x_overlap = static_cast<int>(slice_width  * overlap_w_ratio);
    y_overlap = static_cast<int>(slice_height * overlap_h_ratio);
}

void SahiSlicer::get_resolution_selector(const std::string& res,
                                          int height, int width,
                                          int& x_overlap, int& y_overlap,
                                          int& slice_width, int& slice_height) {
    Orientation orientation = calc_orientation(width, height);
    calc_slice_and_overlap_params(res, height, width, orientation,
                                   x_overlap, y_overlap,
                                   slice_width, slice_height);
}

} // namespace sahi
