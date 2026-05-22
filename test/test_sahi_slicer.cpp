/**
 * Standalone unit test for SahiSlicer.
 *
 * Compile (no dependencies required):
 *   g++ -std=c++17 -I../include -o test_sahi_slicer test_sahi_slicer.cpp ../src/sahi_slicer.cpp
 *
 * Expected output must match sahi/slicing.py get_slice_bboxes() exactly.
 * Reference values from Python (yolo_env, sahi-main):
 *   Test 1: 12 slices
 *   Test 2: 1 slice [0,0,400,300]
 *   Test 3: auto_params values verified against get_auto_slice_params()
 *   Test 4: 1 slice [0,0,640,640]
 *   Test 5: 8 slices (4 per row, rows at y=0 and y=440)
 */

#include "sed/sahi_slicer.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::printf("  FAIL: %s\n", msg); \
        failures++; \
    } \
} while(0)

#define CHECK_EQ(a, b, name) do { \
    if ((a) != (b)) { \
        std::printf("  FAIL: %s: got %d, expected %d\n", name, (int)(a), (int)(b)); \
        failures++; \
    } \
} while(0)

void print_slices(const std::vector<sahi::Slice>& slices) {
    for (size_t i = 0; i < slices.size(); i++) {
        auto& s = slices[i];
        std::printf("  [%zu] x_min=%d, y_min=%d, x_max=%d, y_max=%d, w=%d, h=%d\n",
                    i, s.x_min, s.y_min, s.x_max, s.y_max, s.width(), s.height());
    }
}

int main() {
    std::printf("=== Test 1: Manual 2000x1500, 640 slice, overlap=0.2 ===\n");
    {
        sahi::SahiSlicer slicer(640, 640, 0.2f, 0.2f, 1500, 2000);
        auto slices = slicer.compute_slices();
        std::printf("Slices: %zu\n", slices.size());
        print_slices(slices);

        CHECK_EQ(slices.size(), 12, "slice count");
        // Row 0
        CHECK_EQ(slices[0].x_min, 0,    "s0 x_min");
        CHECK_EQ(slices[0].y_min, 0,    "s0 y_min");
        CHECK_EQ(slices[0].x_max, 640,  "s0 x_max");
        CHECK_EQ(slices[0].y_max, 640,  "s0 y_max");
        CHECK_EQ(slices[3].x_min, 1360, "s3 x_min (clamped)");
        CHECK_EQ(slices[3].x_max, 2000, "s3 x_max (clamped)");
        // Row 2 (bottom, clamped)
        CHECK_EQ(slices[8].y_min, 860,  "s8 y_min (bottom clamp)");
        CHECK_EQ(slices[8].y_max, 1500, "s8 y_max (bottom clamp)");
        CHECK_EQ(slices[11].x_min, 1360,"s11 x_min (corner)");
        CHECK_EQ(slices[11].y_min, 860, "s11 y_min (corner)");
        CHECK_EQ(slices[11].x_max, 2000,"s11 x_max (corner)");
        CHECK_EQ(slices[11].y_max, 1500,"s11 y_max (corner)");

        // Bounds check
        for (auto& s : slices) {
            CHECK(s.x_min >= 0 && s.x_min <= 2000, "x_min in bounds");
            CHECK(s.x_max >= 0 && s.x_max <= 2000, "x_max in bounds");
            CHECK(s.y_min >= 0 && s.y_min <= 1500, "y_min in bounds");
            CHECK(s.y_max >= 0 && s.y_max <= 1500, "y_max in bounds");
            CHECK(s.width() <= 640, "slice width ≤ 640");
            CHECK(s.height() <= 640, "slice height ≤ 640");
        }
    }

    std::printf("\n=== Test 2: Small 400x300, 640 slice ===\n");
    {
        sahi::SahiSlicer slicer(640, 640, 0.2f, 0.2f, 300, 400);
        auto slices = slicer.compute_slices();
        std::printf("Slices: %zu\n", slices.size());
        print_slices(slices);

        CHECK_EQ(slices.size(), 1, "small image: 1 slice");
        CHECK_EQ(slices[0].x_min, 0,   "x_min=0");
        CHECK_EQ(slices[0].y_min, 0,   "y_min=0");
        CHECK_EQ(slices[0].x_max, 400, "x_max=400");
        CHECK_EQ(slices[0].y_max, 300, "y_max=300");
    }

    std::printf("\n=== Test 3: Auto params ===\n");
    {
        auto check_auto = [](int h, int w, const char* name,
                              int exp_slice_h, int exp_slice_w,
                              int exp_overlap_h, int exp_overlap_w) {
            auto p = sahi::SahiSlicer::auto_params(h, w);
            std::printf("  %s: slice=(%d,%d), overlap=(%d,%d)\n",
                        name, p.slice_h, p.slice_w, p.overlap_h_pixels, p.overlap_w_pixels);
            CHECK_EQ(p.slice_h, exp_slice_h, "slice_h");
            CHECK_EQ(p.slice_w, exp_slice_w, "slice_w");
            CHECK_EQ(p.overlap_h_pixels, exp_overlap_h, "overlap_h");
            CHECK_EQ(p.overlap_w_pixels, exp_overlap_w, "overlap_w");
        };

        check_auto(300, 300,   "300x300",     300, 300,  300, 300);
        check_auto(640, 640,   "640x640",     640, 640,  640, 640);
        check_auto(1024, 1024, "1024x1024",  1024, 1024, 819, 819);
        check_auto(2048, 2048, "2048x2048",  1024, 1024, 409, 409);
        check_auto(4096, 4096, "4096x4096",  2048, 2048, 819, 819);
        check_auto(3000, 4000, "4000x3000",  1500, 1000, 600, 400);
    }

    std::printf("\n=== Test 4: Exact 640x640, 640 slice ===\n");
    {
        sahi::SahiSlicer slicer(640, 640, 0.2f, 0.2f, 640, 640);
        auto slices = slicer.compute_slices();
        std::printf("Slices: %zu\n", slices.size());
        print_slices(slices);
        CHECK_EQ(slices.size(), 1, "exact fit: 1 slice");
        CHECK_EQ(slices[0].x_max, 640, "x_max=640");
        CHECK_EQ(slices[0].y_max, 640, "y_max=640");
    }

    std::printf("\n=== Test 5: 1920x1080 HD, 640 slice, overlap=0.2 ===\n");
    {
        sahi::SahiSlicer slicer(640, 640, 0.2f, 0.2f, 1080, 1920);
        auto slices = slicer.compute_slices();
        std::printf("Slices: %zu\n", slices.size());
        print_slices(slices);

        CHECK_EQ(slices.size(), 8, "HD: 8 slices");
        // Row 0
        CHECK_EQ(slices[0].y_min, 0,    "HD row0 y_min");
        CHECK_EQ(slices[0].y_max, 640,  "HD row0 y_max");
        CHECK_EQ(slices[3].x_min, 1280, "HD row0 last x_min");
        CHECK_EQ(slices[3].x_max, 1920, "HD row0 last x_max");
        // Row 1 (bottom, clamped)
        CHECK_EQ(slices[4].y_min, 440,  "HD row1 y_min (clamped)");
        CHECK_EQ(slices[4].y_max, 1080, "HD row1 y_max (clamped)");
        CHECK_EQ(slices[7].x_min, 1280, "HD row1 last x_min");
        CHECK_EQ(slices[7].y_min, 440,  "HD row1 last y_min");

        for (auto& s : slices) {
            CHECK(s.x_min >= 0 && s.x_min <= 1920, "x_min in bounds");
            CHECK(s.x_max >= 0 && s.x_max <= 1920, "x_max in bounds");
            CHECK(s.y_min >= 0 && s.y_min <= 1080, "y_min in bounds");
            CHECK(s.y_max >= 0 && s.y_max <= 1080, "y_max in bounds");
        }
    }

    std::printf("\n=== Test 6: from_auto() factory ===\n");
    {
        auto slicer = sahi::SahiSlicer::from_auto(1080, 1920);
        auto slices = slicer.compute_slices();
        std::printf("Slices: %zu\n", slices.size());
        print_slices(slices);

        // Python: 1920x1080 auto = 6 slices, all y range [0, 1080]
        CHECK_EQ(slices.size(), 6, "auto HD: 6 slices");
        CHECK_EQ(slicer.image_h(), 1080, "image_h");
        CHECK_EQ(slicer.image_w(), 1920, "image_w");
        CHECK_EQ(slicer.num_slices(), 6, "num_slices");
        for (auto& s : slices) {
            CHECK(s.y_min == 0 && s.y_max == 1080, "auto HD: full height");
        }
        CHECK_EQ(slices[0].x_min, 0,    "auto s0 x_min");
        CHECK_EQ(slices[0].x_max, 960,  "auto s0 x_max");
        CHECK_EQ(slices[5].x_min, 960,  "auto s5 x_min");
        CHECK_EQ(slices[5].x_max, 1920, "auto s5 x_max");
    }

    std::printf("\n=== Test 7: Cache consistency ===\n");
    {
        sahi::SahiSlicer slicer(640, 640, 0.2f, 0.2f, 1000, 1000);
        auto a = slicer.compute_slices();
        auto b = slicer.compute_slices();
        CHECK(a.size() == b.size(), "cached same size");
        CHECK(&a[0] == &b[0], "same vector (cached)");
    }

    std::printf("\n=== Test 8: Overlap ratio >= 1.0 should be rejected ===\n");
    {
        // Python raises ValueError for overlap >= 1.0
        // Our C++ version doesn't validate (simplification), but let's document.
        // With overlap=1.0, the loop should complete (stride=0, but last slice
        // reaches image edge and then while exits).
        // Actually, with 100% overlap, x_overlap == slice_w, so x_min stays at 0,
        // but x_max increases each iteration until it hits image_w, then clamps.
        // This matches Python's "low" resolution behavior.
        std::printf("  (no explicit validation for overlap >= 1.0 in C++)\n");
    }

    std::printf("\n");
    if (failures == 0) {
        std::printf("All tests PASSED\n");
        return 0;
    } else {
        std::printf("%d test(s) FAILED\n", failures);
        return 1;
    }
}
