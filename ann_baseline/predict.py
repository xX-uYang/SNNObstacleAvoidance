from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from PIL import Image

from ann_baseline.checkpoint import load_checkpoint
from ann_baseline.constants import CLASS_NAMES
from ann_baseline.data import build_transform
from ann_baseline.interfaces import SNNPrior


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ANN inference on one image.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--snn-left", type=float)
    parser.add_argument("--snn-straight", type=float)
    parser.add_argument("--snn-right", type=float)
    parser.add_argument(
        "--snn-label", choices=("left", "straight", "right")
    )
    parser.add_argument(
        "--max-steer-degrees",
        type=float,
        default=30.0,
        help="Only used to convert normalized steering output to an approximate angle.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    model, payload = load_checkpoint(args.checkpoint, device)
    transform = build_transform(int(payload["input_size"]), training=False)
    image = transform(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)

    if args.snn_label is not None:
        if (
            args.snn_left is not None
            or args.snn_straight is not None
            or args.snn_right is not None
        ):
            raise ValueError("Use either --snn-label or probabilities, not both.")
        prior = SNNPrior.from_label(args.snn_label)
    else:
        prior = SNNPrior.from_probabilities(
            args.snn_left, args.snn_straight, args.snn_right
        )
    prior_tensor = torch.tensor(
        [prior.as_model_input()],
        dtype=torch.float32,
        device=device,
    )

    start = time.perf_counter()
    with torch.inference_mode():
        output = model(image, prior_tensor if model.use_snn_prior else None)
        if device.type == "cuda":
            torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000.0

    result = {
        "image": str(args.image.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "task": model.task,
        "used_snn_prior": model.use_snn_prior,
        "snn_prior": {
            "left": prior.left_probability,
            "straight": prior.straight_probability,
            "right": prior.right_probability,
            "available": prior.available,
        },
        "ann_inference_latency_ms": latency_ms,
    }
    if model.task == "classification":
        probabilities = torch.softmax(output, dim=1)[0].cpu().tolist()
        predicted_index = int(torch.argmax(output, dim=1).item())
        result.update(
            {
                "direction": CLASS_NAMES[predicted_index],
                "probabilities": {
                    "left": probabilities[0],
                    "straight": probabilities[1],
                    "right": probabilities[2],
                },
            }
        )
    else:
        normalized = float(output[0, 0].item())
        result.update(
            {
                "steering_normalized": normalized,
                "steering_degrees_approx": normalized * args.max_steer_degrees,
            }
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
