"""Export the Yoga-16 model to ONNX and verify it with ONNX Runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from model import AttentionYogaNODE, ONNXYogaNODE, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/MatNODE_Yoga16.pth")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/MatNODE_Yoga16.onnx")
    )
    parser.add_argument("--rk4-steps", type=int, default=8)
    args = parser.parse_args()

    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    labels = raw.get("labels") if isinstance(raw, dict) else None
    num_classes = len(labels) if labels else 16
    source = AttentionYogaNODE(num_classes=num_classes)
    load_checkpoint(source, args.checkpoint)
    export_model = ONNXYogaNODE(source.eval(), rk4_steps=args.rk4_steps).eval()
    dummy = torch.randn(1, 16, 212)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        export_model,
        dummy,
        args.output,
        input_names=["pose_features"],
        output_names=["logits"],
        dynamic_axes={"pose_features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(
        str(args.output), providers=["CPUExecutionProvider"]
    )
    with torch.inference_mode():
        expected = export_model(dummy).numpy()
    actual = session.run(None, {"pose_features": dummy.numpy()})[0]
    verification = {
        "onnx_path": str(args.output),
        "size_bytes": args.output.stat().st_size,
        "opset": 17,
        "rk4_steps": args.rk4_steps,
        "max_absolute_error": float(np.max(np.abs(expected - actual))),
        "verified": bool(np.allclose(expected, actual, rtol=1e-3, atol=1e-4)),
        "note": "ONNX uses fixed-step RK4 because adaptive torchdiffeq is not portable.",
    }
    args.output.with_suffix(".onnx.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
