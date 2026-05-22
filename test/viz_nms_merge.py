#!/usr/bin/env python3
"""viz_nms_merge.py — visualize full vs slice detections + NMS merge

Generates side-by-side comparison: full dets | slice dets | merged result
Slice size < image size forces real multi-slice, showing the difference.

Usage:
    python3 viz_nms_merge.py --images img1.jpg img2.jpg --output /tmp/nms_viz
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


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
    blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
    return blob, gain, pad_w, pad_h


def decode_raw(raw_output, gain, pad_w, pad_h, orig_w, orig_h, conf_thresh):
    boxes = []
    for a in range(raw_output.shape[0]):
        x1, y1 = raw_output[a, 0], raw_output[a, 1]
        x2, y2 = raw_output[a, 2], raw_output[a, 3]
        conf = float(raw_output[a, 4])
        cls_id = int(raw_output[a, 5])
        if conf < conf_thresh or cls_id < 0:
            continue
        inv = 1.0 / gain
        x1 = max(0.0, min((x1 - pad_w) * inv, float(orig_w)))
        y1 = max(0.0, min((y1 - pad_h) * inv, float(orig_h)))
        x2 = max(0.0, min((x2 - pad_w) * inv, float(orig_w)))
        y2 = max(0.0, min((y2 - pad_h) * inv, float(orig_h)))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                       "confidence": conf, "class_id": cls_id})
    return boxes


def compute_iou(a, b):
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    aa = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    ab = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return inter / (aa + ab - inter)


def nms_per_class(boxes, iou_thresh):
    if not boxes:
        return []
    by_class = {}
    for i, b in enumerate(boxes):
        by_class.setdefault(b["class_id"], []).append(i)
    keep = [True] * len(boxes)
    for indices in by_class.values():
        indices.sort(key=lambda i: boxes[i]["confidence"], reverse=True)
        for ii in range(len(indices)):
            idx_i = indices[ii]
            if not keep[idx_i]:
                continue
            for jj in range(ii + 1, len(indices)):
                idx_j = indices[jj]
                if not keep[idx_j]:
                    continue
                if compute_iou(boxes[idx_i], boxes[idx_j]) > iou_thresh:
                    keep[idx_j] = False
    return [b for b, k in zip(boxes, keep) if k]


COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]


def draw_boxes(img, boxes, color, label_prefix="", thickness=2):
    for b in boxes:
        x1, y1 = int(b["x1"]), int(b["y1"])
        x2, y2 = int(b["x2"]), int(b["y2"])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        name = COCO_CLASSES[b["class_id"]] if b["class_id"] < 80 else f"cls{b['class_id']}"
        label = f"{label_prefix}{name} {b['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(img, label, (x1 + 1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def draw_slice_grid(img, slices):
    for (sx1, sy1, sx2, sy2) in slices:
        cv2.rectangle(img, (sx1, sy1), (sx2, sy2), (128, 128, 128), 1)
        cv2.putText(img, f"[{sx1},{sy1}]", (sx1 + 2, sy1 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)


def hstack_with_label(panels, labels, panel_h=30):
    """Stack panels horizontally with title bars."""
    n = len(panels)
    max_h = max(p.shape[0] for p in panels)
    width = sum(p.shape[1] for p in panels)
    result = np.zeros((max_h + panel_h, width, 3), dtype=np.uint8)
    x = 0
    for i, panel in enumerate(panels):
        h, w = panel.shape[:2]
        result[panel_h:panel_h + h, x:x + w] = panel
        # Title bar
        cv2.rectangle(result, (x, 0), (x + w, panel_h), (40, 40, 40), -1)
        cv2.putText(result, labels[i], (x + 5, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        x += w
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, action="append",
                        dest="images", help="Image path (repeatable)")
    parser.add_argument("--onnx", default="config/yolov8n.onnx")
    parser.add_argument("--output", default="/tmp/nms_viz")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--slice-size", type=int, default=320)
    parser.add_argument("--overlap", type=float, default=0.3)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    session = ort.InferenceSession(args.onnx)
    input_name = session.get_inputs()[0].name

    for img_path in args.images:
        basename = Path(img_path).stem
        img = cv2.imread(img_path)
        if img is None:
            print(f"SKIP: cannot read {img_path}")
            continue
        orig_h, orig_w = img.shape[:2]
        print(f"{basename} ({orig_w}x{orig_h})")

        # Full-image inference
        blob, gain, pad_w, pad_h = preprocess(img)
        raw = session.run(None, {input_name: blob})[0][0]
        full_dets = nms_per_class(
            decode_raw(raw, gain, pad_w, pad_h, orig_w, orig_h, args.conf),
            0.45)

        # SAHI slices
        slices = get_slice_bboxes(orig_h, orig_w,
                                  args.slice_size, args.slice_size,
                                  args.overlap, args.overlap)
        print(f"  Slices: {len(slices)}  "
              f"({args.slice_size}x{args.slice_size}, overlap={args.overlap})")

        slice_dets = []
        for (sx1, sy1, sx2, sy2) in slices:
            roi = img[sy1:sy2, sx1:sx2]
            s_blob, s_gain, s_pad_w, s_pad_h = preprocess(roi)
            s_raw = session.run(None, {input_name: s_blob})[0][0]
            s_dets = nms_per_class(
                decode_raw(s_raw, s_gain, s_pad_w, s_pad_h,
                           sx2 - sx1, sy2 - sy1, args.conf),
                0.45)
            for d in s_dets:
                d["x1"] += sx1; d["y1"] += sy1
                d["x2"] += sx1; d["y2"] += sy1
            slice_dets.extend(s_dets)

        # Merge
        merged = nms_per_class(full_dets + slice_dets, args.nms_iou)

        # Tag merged boxes by source
        for b in merged:
            matched_full = any(
                fb["class_id"] == b["class_id"] and
                abs(fb["confidence"] - b["confidence"]) < 0.005 and
                abs(fb["x1"] - b["x1"]) < 1.0 and
                abs(fb["y1"] - b["y1"]) < 1.0
                for fb in full_dets)
            b["from_full"] = matched_full

        # ── Build panels ──
        # Panel 1: Full-image detections (green)
        p1 = img.copy()
        draw_boxes(p1, full_dets, (0, 200, 0), "F:")

        # Panel 2: Slice detections (blue) + slice grid
        p2 = img.copy()
        draw_slice_grid(p2, slices)
        draw_boxes(p2, slice_dets, (200, 120, 0), "S:")

        # Panel 3: Merged result — red=from_full, yellow=from_slice
        p3 = img.copy()
        full_kept = [b for b in merged if b.get("from_full")]
        slice_kept = [b for b in merged if not b.get("from_full")]
        draw_boxes(p3, full_kept, (0, 200, 0), "F:")
        draw_boxes(p3, slice_kept, (0, 140, 255), "S:")

        # Compose
        composite = hstack_with_label(
            [p1, p2, p3],
            [f"Full-image ({len(full_dets)} dets)",
             f"SAHI slices ({len(slice_dets)} dets, {len(slices)} slices)",
             f"Merged ({len(merged)}: green=full {len(full_kept)}, orange=slice {len(slice_kept)})"])

        # Scale if too wide
        if composite.shape[1] > 1800:
            scale = 1800.0 / composite.shape[1]
            composite = cv2.resize(composite, (0, 0), fx=scale, fy=scale)

        out_path = os.path.join(args.output, f"{basename}_viz.jpg")
        cv2.imwrite(out_path, composite)
        print(f"  → {out_path}\n"
              f"     Full={len(full_dets)}  Slice={len(slice_dets)}  "
              f"Merged={len(merged)}  "
              f"(from_full={len(full_kept)} from_slice={len(slice_kept)})")

    print(f"\nDone — {len(args.images)} images in {args.output}")


if __name__ == "__main__":
    main()
