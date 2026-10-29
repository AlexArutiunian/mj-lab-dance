#!/usr/bin/env python3
"""Export ONNX feed-forward graphs to a compact CPU-only JS format.

The ONNX files remain the source of truth. We retain the full graph topology,
float32 weights and operator attributes for browser inference without ORT WASM.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

SUPPORTED = {
    "Gemm", "MatMul", "Add", "Sub", "Mul", "Div",
    "Relu", "Tanh", "Sigmoid", "Elu", "LeakyRelu",
    "Identity", "Cast", "Reshape", "Flatten",
    "Squeeze", "Unsqueeze", "Transpose", "Concat",
    "Neg", "Exp", "Sqrt", "Pow", "Clip", "Constant",
}

def export_policy(path):
    path = Path(path)
    graph = onnx.load(str(path), load_external_data=True).graph
    operators = Counter(node.op_type for node in graph.node)
    unsupported = sorted(set(operators) - SUPPORTED)
    print("JS export", path, "operators", dict(operators), flush=True)
    if unsupported:
        raise ValueError(f"Unsupported ONNX operators in {path}: {unsupported}")
    arrays = {}
    for tensor in graph.initializer:
        arrays[tensor.name] = numpy_helper.to_array(tensor).copy()
    nodes = []
    for node in graph.node:
        attrs = {}
        for attr in node.attribute:
            value = helper.get_attribute_value(attr)
            if attr.name == "value":
                arrays[node.output[0]] = numpy_helper.to_array(value).copy()
                continue
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            elif isinstance(value, (list, tuple)):
                value = [v.decode("utf-8") if isinstance(v, bytes) else v for v in value]
            elif isinstance(value, np.generic):
                value = value.item()
            if not isinstance(value, (str, int, float, list)):
                raise ValueError(f"Unsupported ONNX attribute {node.op_type}.{attr.name}: {type(value)}")
            attrs[attr.name] = value
        if node.op_type != "Constant":
            nodes.append({"op":node.op_type, "inputs":list(node.input), "outputs":list(node.output), "attrs":attrs})
    floats = []
    constants = {}
    for name, data in arrays.items():
        data = np.asarray(data)
        if data.dtype.kind not in "fiub":
            raise ValueError(f"Unsupported tensor {name}: {data.dtype}")
        # Preserve dimensions and exact fp32 parameter values.
        arr = np.asarray(data, dtype=np.float32).ravel()
        constants[name] = {"shape":list(data.shape), "offset":len(floats), "length":len(arr)}
        floats.extend(arr)
    raw = np.asarray(floats, dtype="<f4")
    path.with_name("policy.js.bin").write_bytes(raw.tobytes())
    payload = {
        "format":"onnx-fp32-graph-v1",
        "inputs":[x.name for x in graph.input if x.name not in arrays],
        "outputs":[x.name for x in graph.output],
        "constants":constants,
        "nodes":nodes,
        "float_count":len(raw),
    }
    if len(payload["inputs"])!=1 or len(payload["outputs"])!=1:
        raise ValueError(f"Expected 1 input, 1 output, got {payload['inputs']} {payload['outputs']}")
    path.with_name("policy.js.json").write_text(json.dumps(payload,separators=(",",":")), encoding="utf-8")
    print("JS export complete", path, len(nodes), "ops", len(raw), "floats", flush=True)
