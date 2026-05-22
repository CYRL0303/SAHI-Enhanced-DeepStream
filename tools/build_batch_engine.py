#!/usr/bin/env python3
"""build_batch_engine.py — Build dynamic-batch FP16 TensorRT engine from ONNX.

Usage:
    python3 build_batch_engine.py yolov8n.onnx yolov8n_batch.engine [min_batch] [opt_batch] [max_batch]

Default batch profile: min=1, opt=4, max=8 (optimized for SAHI slicing).
"""

import sys
import os

import tensorrt as trt


def build_engine(onnx_path, out_path, min_batch=1, opt_batch=4, max_batch=8):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2 GiB
    config.set_flag(trt.BuilderFlag.FP16)

    # Check if FP16 is supported
    if not builder.platform_has_fast_fp16:
        print("WARNING: FP16 not supported on this platform, falling back to FP32")

    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, 'rb') as f:
        data = f.read()
        if not parser.parse(data):
            print("ERROR: Failed to parse ONNX model")
            for i in range(parser.num_errors):
                print(f"  {parser.get_error(i)}")
            sys.exit(1)

    print(f"ONNX parsed OK: {len(data)} bytes, {network.num_layers} layers")

    # Print network I/O
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"  Input:  {inp.name} shape={inp.shape}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"  Output: {out.name} shape={out.shape}")

    # Optimization profile: dynamic batch
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "input",
        (min_batch, 3, 640, 640),
        (opt_batch, 3, 640, 640),
        (max_batch, 3, 640, 640))
    config.add_optimization_profile(profile)

    print(f"Building engine with batch range [{min_batch}, {opt_batch}, {max_batch}]...")
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        print("ERROR: build_serialized_network returned None")
        sys.exit(1)

    with open(out_path, 'wb') as f:
        f.write(plan)

    print(f"Engine saved: {out_path} ({len(plan):,} bytes)")
    print(f"Batch range: [{min_batch}, {opt_batch}, {max_batch}]")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    onnx_file = sys.argv[1]
    out_file = sys.argv[2]
    min_b = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    opt_b = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    max_b = int(sys.argv[5]) if len(sys.argv) > 5 else 8

    sys.exit(build_engine(onnx_file, out_file, min_b, opt_b, max_b))
