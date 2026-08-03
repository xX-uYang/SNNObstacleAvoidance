from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from ann_baseline.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    model, payload = load_checkpoint(args.checkpoint, device)
    input_size = int(payload["input_size"])
    image = torch.zeros(1, 3, input_size, input_size, dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if model.use_snn_prior:
        prior = torch.tensor(
            [[1 / 3, 1 / 3, 1 / 3, 0.0]], dtype=torch.float32
        )
        torch.onnx.export(
            model,
            (image, prior),
            args.output,
            opset_version=args.opset,
            input_names=["image", "snn_prior"],
            output_names=["output"],
            dynamic_axes={
                "image": {0: "batch"},
                "snn_prior": {0: "batch"},
                "output": {0: "batch"},
            },
            do_constant_folding=True,
        )
    else:
        torch.onnx.export(
            model,
            image,
            args.output,
            opset_version=args.opset,
            input_names=["image"],
            output_names=["output"],
            dynamic_axes={"image": {0: "batch"}, "output": {0: "batch"}},
            do_constant_folding=True,
        )
    print(str(args.output.resolve()))


if __name__ == "__main__":
    main()
