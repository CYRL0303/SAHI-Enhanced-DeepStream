#pragma once

/// Pure C++ slicing engine ported from sahi/slicing.py.
/// Zero external dependencies — no OpenCV, no ROS, no CUDA.

#include <string>
#include <vector>

namespace sahi {

struct Slice {
    int x_min, y_min, x_max, y_max;

    int width() const { return x_max - x_min; }
    int height() const { return y_max - y_min; }
};

struct AutoParams {
    int slice_h, slice_w;
    int overlap_h_pixels, overlap_w_pixels;
};

class SahiSlicer {
public:
    /// Manual mode: slice size + overlap ratios (0.0 ~ 0.99).
    /// Pixel overlaps are computed as int(ratio * slice_size), matching Python.
    SahiSlicer(int slice_h, int slice_w,
               float overlap_h_ratio, float overlap_w_ratio,
               int image_h, int image_w);

    /// Auto-resolution factory.
    /// Determines slice count and overlap from image total pixels.
    static SahiSlicer from_auto(int image_h, int image_w);

    /// Compute all slice coordinates. Result is cached after first call.
    std::vector<Slice> compute_slices() const;

    int image_h() const { return image_h_; }
    int image_w() const { return image_w_; }
    int num_slices() const;

    /// Public for unit testing. Returns precomputed pixel overlaps.
    static AutoParams auto_params(int image_h, int image_w);

private:
    // Tagged constructor with precomputed pixel overlaps (from_auto path)
    SahiSlicer(int slice_h, int slice_w,
               int overlap_h_pixels, int overlap_w_pixels,
               int image_h, int image_w, bool);

    // --- Static helpers ported from sahi/slicing.py ---

    /// floor(log2(resolution))
    static int calc_resolution_factor(int resolution);

    enum class Orientation { kVertical, kHorizontal, kSquare };
    static Orientation calc_orientation(int width, int height);

    /// Map orientation + tier (slide/ratio) → (slice_row, slice_col, ratios).
    static void calc_ratio_and_slice(Orientation orientation, int slide, float ratio,
                                     int& slice_row, int& slice_col,
                                     float& overlap_h_ratio, float& overlap_w_ratio);

    /// Map resolution tier string → concrete pixel overlaps and slice dims.
    static void calc_slice_and_overlap_params(const std::string& resolution,
                                              int height, int width,
                                              Orientation orientation,
                                              int& x_overlap, int& y_overlap,
                                              int& slice_width, int& slice_height);

    /// Thin orchestrator: orientation + tier → params.
    static void get_resolution_selector(const std::string& res, int height, int width,
                                        int& x_overlap, int& y_overlap,
                                        int& slice_width, int& slice_height);

    int slice_h_, slice_w_;
    int overlap_h_pixels_, overlap_w_pixels_;
    int image_h_, image_w_;

    mutable std::vector<Slice> cached_slices_;
    mutable bool slices_computed_ = false;
};

} // namespace sahi
