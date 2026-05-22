#!/usr/bin/env python3
"""verify_nms_merge.py — Step 3: verify C++ sahi_nms_merge() matches Python NMS merge

For 50-60 images from the fervent-eagle-coco dataset:
  1. Python full-image ONNX inference → full_dets
  2. Python SAHI slicing + per-slice ONNX → slice_dets (translated to image coords)
  3. Python merge(full_dets, slice_dets) → merged_py (reference)
  4. Write full_dets + slice_dets as JSON → C++ test_nms reads them → merged_cpp
  5. Compare merged_cpp vs merged_py per image

Usage:
    python3 verify_nms_merge.py --num 60 [--iou 0.5] [--conf 0.25]
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# ===================================================================
# SAHI slicing (port from sahi/slicing.py)
# ===================================================================
def get_slice_bboxes(image_h, image_w, slice_h, slice_w,
                     overlap_h_ratio, overlap_w_ratio):
    slices = []
    overlap_h_px = int(slice_h * overlap_h_ratio)
    overlap_w_px = int(slice_w * overlap_w_ratio)
    y_max, y_min = 0, 0
    while y_max < image_h:
        x_max, x_min = 0, 0
        y_max = y_min + slice_h
        while x_max < image_w:
            x_max = x_min + slice_w
            if y_max > image_h or x_max > image_w:
                x_max = min(image_w, x_max)
                y_max = min(image_h, y_max)
                x_min = max(0, x_max - slice_w)
                y_min = max(0, y_max - slice_h)
            slices.append([x_min, y_min, x_max, y_max])
            x_min = x_max - overlap_w_px
        y_min = y_max - overlap_h_px
    return slices


# ===================================================================
# ONNX preprocessing (must match C++ TrtYoloInfer::preprocess exactly)
# ===================================================================
def preprocess(img, net_w=640, net_h=640):
    h, w = img.shape[:2]
    gain = min(net_w / w, net_h / h)
    new_w, new_h = int(w * gain), int(h * gain)
    pad_w = (net_w - new_w) // 2
    pad_h = (net_h - new_h) // 2

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    letterbox = np.full((net_h, net_w, 3), 114, dtype=np.uint8)
    letterbox[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    rgb = letterbox[:, :, ::-1]
    blob = rgb.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)
    blob = np.expand_dims(blob, 0)
    return blob, gain, pad_w, pad_h


# ===================================================================
# Decode ONNX raw output [1, 8400, 6] → boxes in image coords
# ===================================================================
def decode_raw(raw_output, gain, pad_w, pad_h, orig_w, orig_h, conf_thresh):
    """raw_output shape: [8400, 6] = [x1, y1, x2, y2, conf, cls_id]"""
    boxes = []
    for a in range(raw_output.shape[0]):
        x1 = raw_output[a, 0]
        y1 = raw_output[a, 1]
        x2 = raw_output[a, 2]
        y2 = raw_output[a, 3]
        conf = float(raw_output[a, 4])
        cls_id = int(raw_output[a, 5])

        if conf < conf_thresh:
            continue
        if cls_id < 0:
            continue

        inv_gain = 1.0 / gain
        x1 = (x1 - pad_w) * inv_gain
        y1 = (y1 - pad_h) * inv_gain
        x2 = (x2 - pad_w) * inv_gain
        y2 = (y2 - pad_h) * inv_gain

        x1 = max(0.0, min(x1, float(orig_w)))
        y1 = max(0.0, min(y1, float(orig_h)))
        x2 = max(0.0, min(x2, float(orig_w)))
        y2 = max(0.0, min(y2, float(orig_h)))

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append({
            "x1": float(x1), "y1": float(y1),
            "x2": float(x2), "y2": float(y2),
            "confidence": conf, "class_id": cls_id
        })
    return boxes


# ===================================================================
# Python NMS merge (reference implementation — must match C++ sahi_nms_merge)
# ===================================================================
def compute_iou(a, b):
    inter_x1 = max(a["x1"], b["x1"])
    inter_y1 = max(a["y1"], b["y1"])
    inter_x2 = min(a["x2"], b["x2"])
    inter_y2 = min(a["y2"], b["y2"])
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return inter / (area_a + area_b - inter)


def py_nms_per_class(boxes, iou_thresh):
    """Per-class NMS: identical logic to TrtYoloInfer::nms_per_class()"""
    if not boxes:
        return []

    # Group by class_id
    by_class = {}
    for i, b in enumerate(boxes):
        cls = b["class_id"]
        by_class.setdefault(cls, []).append(i)

    keep = [True] * len(boxes)

    for cls, indices in by_class.items():
        indices.sort(key=lambda i: boxes[i]["confidence"], reverse=True)

        for ii in range(len(indices)):
            idx_i = indices[ii]
            if not keep[idx_i]:
                continue
            for jj in range(ii + 1, len(indices)):
                idx_j = indices[jj]
                if not keep[idx_j]:
                    continue
                iou = compute_iou(boxes[idx_i], boxes[idx_j])
                if iou > iou_thresh:
                    keep[idx_j] = False

    return [b for b, k in zip(boxes, keep) if k]


def py_nms_merge(full_dets, slice_dets, iou_thresh):
    """Python equivalent of sahi::sahi_nms_merge()"""
    merged = full_dets + slice_dets
    return py_nms_per_class(merged, iou_thresh)


# ===================================================================
# JSON I/O
# ===================================================================
def write_boxes_json(path, boxes):
    with open(path, 'w') as f:
        f.write('{\n  "boxes": [\n')
        for i, b in enumerate(boxes):
            comma = "," if i + 1 < len(boxes) else ""
            f.write(
                f'    {{"x1":{b["x1"]},"y1":{b["y1"]},'
                f'"x2":{b["x2"]},"y2":{b["y2"]},'
                f'"confidence":{b["confidence"]},"class_id":{b["class_id"]}'
                f'}}{comma}\n'
            )
        f.write('  ]\n}\n')


def read_boxes_json(path):
    with open(path) as f:
        data = json.load(f)
    return data["boxes"]


# ===================================================================
# Comparison
# ===================================================================
def center_dist(a, b):
    cx_a = (a["x1"] + a["x2"]) / 2
    cy_a = (a["y1"] + a["y2"]) / 2
    cx_b = (b["x1"] + b["x2"]) / 2
    cy_b = (b["y1"] + b["y2"]) / 2
    return np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)


def compare_boxes(py_boxes, cpp_boxes, center_tol=15.0, conf_tol=0.02):
    """Match boxes by nearest center distance within same class."""
    errors = []

    if len(py_boxes) != len(cpp_boxes):
        errors.append(
            f"count: py={len(py_boxes)} cpp={len(cpp_boxes)}"
        )
        return errors

    # Build match index: for each py box, find nearest cpp box of same class
    matched_cpp = set()
    for i, pb in enumerate(py_boxes):
        best_dist = float('inf')
        best_j = -1
        for j, cb in enumerate(cpp_boxes):
            if j in matched_cpp:
                continue
            if cb["class_id"] != pb["class_id"]:
                continue
            d = center_dist(pb, cb)
            if d < best_dist:
                best_dist = d
                best_j = j

        if best_j < 0:
            errors.append(f"box[{i}]: no matching class {pb['class_id']} in C++")
            continue
        if best_dist > center_tol:
            errors.append(
                f"box[{i}]: center dist {best_dist:.1f} > {center_tol}"
            )
            continue

        cb = cpp_boxes[best_j]
        conf_diff = abs(pb["confidence"] - cb["confidence"])
        if conf_diff > conf_tol:
            errors.append(
                f"box[{i}]: conf diff {conf_diff:.4f} py={pb['confidence']:.4f} "
                f"cpp={cb['confidence']:.4f}"
            )
            continue

        matched_cpp.add(best_j)

    return errors


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Verify C++ sahi_nms_merge() against Python reference"
    )
    parser.add_argument("--data-dir",
                        default="/home/jetson/Desktop/fervent-eagle-coco",
                        help="Path to COCO dataset")
    parser.add_argument("--onnx",
                        default="config/yolov8n.onnx",
                        help="Path to ONNX model")
    parser.add_argument("--test-nms",
                        default="build/test_nms",
                        help="Path to C++ test_nms executable")
    parser.add_argument("--output-dir",
                        default="/tmp/nms_verify",
                        help="Output directory for JSON files")
    parser.add_argument("--num", type=int, default=60,
                        help="Number of images to verify")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="NMS merge IoU threshold")
    parser.add_argument("--slice-size", type=int, default=640,
                        help="SAHI slice size")
    parser.add_argument("--overlap", type=float, default=0.2,
                        help="SAHI overlap ratio")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load ONNX session ──
    print(f"Loading ONNX: {args.onnx}")
    session = ort.InferenceSession(args.onnx)
    input_name = session.get_inputs()[0].name
    print(f"Input name: {input_name}")

    # ── Find images ──
    images_dir = os.path.join(args.data_dir, "images", "train")
    labels_dir = os.path.join(args.data_dir, "labels", "train")

    image_files = sorted(
        p for p in Path(images_dir).glob("*")
        if p.suffix.lower() in ('.jpg', '.jpeg', '.png')
    )
    available = []
    for ip in image_files:
        lbl = Path(labels_dir) / (ip.stem + ".txt")
        if lbl.exists():
            available.append(str(ip))

    # Select images (stratified by detection count from prior runs if possible,
    # otherwise just take the first N)
    selected = available[:args.num]
    print(f"Selected: {len(selected)} images out of {len(available)} available")

    # ── Process each image ──
    passed = 0
    failed = 0
    no_det = 0
    failed_images = []

    for idx, img_path in enumerate(selected):
        basename = os.path.basename(img_path)
        stem = Path(img_path).stem
        print(f"\n[{idx + 1}/{len(selected)}] {basename} ...", end=" ", flush=True)

        img = cv2.imread(img_path)
        if img is None:
            print("SKIP (cannot read)")
            continue
        orig_h, orig_w = img.shape[:2]

        # ── 1. Full-image ONNX inference ──
        blob, gain, pad_w, pad_h = preprocess(img)
        onnx_out = session.run(None, {input_name: blob})
        raw = onnx_out[0]  # [1, 8400, 6]
        full_dets = decode_raw(raw[0], gain, pad_w, pad_h,
                               orig_w, orig_h, args.conf)
        # Apply per-class NMS (single-source)
        full_dets = py_nms_per_class(full_dets, 0.45)

        # ── 2. SAHI slicing + per-slice ONNX ──
        slices = get_slice_bboxes(orig_h, orig_w,
                                  args.slice_size, args.slice_size,
                                  args.overlap, args.overlap)
        slice_dets = []
        for (sx1, sy1, sx2, sy2) in slices:
            roi = img[sy1:sy2, sx1:sx2]
            s_blob, s_gain, s_pad_w, s_pad_h = preprocess(roi)
            s_out = session.run(None, {input_name: s_blob})
            s_raw = s_out[0]
            s_dets = decode_raw(s_raw[0], s_gain, s_pad_w, s_pad_h,
                                sx2 - sx1, sy2 - sy1, args.conf)
            s_dets = py_nms_per_class(s_dets, 0.45)
            # Translate slice coords → full-image coords
            for d in s_dets:
                d["x1"] += sx1
                d["y1"] += sy1
                d["x2"] += sx1
                d["y2"] += sy1
            slice_dets.extend(s_dets)

        # ── 3. Python merge (reference) ──
        merged_py = py_nms_merge(full_dets, slice_dets, args.iou)

        if len(full_dets) == 0 and len(slice_dets) == 0:
            no_det += 1
            print(f"0 det (full=0 slice=0) — skip (no boxes to merge)")
            continue

        print(f"full={len(full_dets)} slice={len(slice_dets)} "
              f"merged_py={len(merged_py)}", end=" ", flush=True)

        # ── 4. Write JSON + run C++ merge ──
        full_json = os.path.join(args.output_dir, f"{stem}_full.json")
        slice_json = os.path.join(args.output_dir, f"{stem}_slice.json")
        merged_json = os.path.join(args.output_dir, f"{stem}_merged_cpp.json")

        write_boxes_json(full_json, full_dets)
        write_boxes_json(slice_json, slice_dets)

        cmd = [
            args.test_nms,
            full_json, slice_json, merged_json,
            str(args.iou)
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                print(f"FAIL (C++ crash)\n{result.stderr[:200]}")
                failed += 1
                failed_images.append((basename, "C++ crash"))
                continue
        except subprocess.TimeoutExpired:
            print("FAIL (C++ timeout)")
            failed += 1
            failed_images.append((basename, "C++ timeout"))
            continue
        except FileNotFoundError:
            print(f"FAIL (test_nms not found at {args.test_nms})")
            sys.exit(1)

        # ── 5. Compare ──
        try:
            merged_cpp = read_boxes_json(merged_json)
        except Exception as e:
            print(f"FAIL (JSON read: {e})")
            failed += 1
            failed_images.append((basename, f"JSON: {e}"))
            continue

        errors = compare_boxes(merged_py, merged_cpp)
        if errors:
            print(f"FAIL ({len(errors)} mismatches)")
            for e in errors[:3]:
                print(f"       {e}")
            failed += 1
            failed_images.append((basename, "; ".join(errors[:3])))
        else:
            print("PASS")
            passed += 1

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"Total: {len(selected)} | Pass: {passed} | Fail: {failed} | "
          f"No-det: {no_det}")
    if failed_images:
        print(f"\nFailed images ({len(failed_images)}):")
        for name, reason in failed_images[:20]:
            print(f"  {name}: {reason}")
    print(f"\n{'ALL PASSED!' if failed == 0 else 'SOME FAILED'}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
