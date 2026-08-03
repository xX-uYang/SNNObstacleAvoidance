from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from PIL import Image

from ann_baseline.checkpoint import load_checkpoint
from ann_baseline.data import build_transform
from ann_baseline.interfaces import SNNPrior


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark warm batch-one ANN model inference latency."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    return parser.parse_args()


def percentile(values: List[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("--warmup must be non-negative and --repeats positive.")
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
    neutral = SNNPrior().as_model_input()
    prior = torch.tensor([neutral], dtype=torch.float32, device=device)

    def run_once() -> None:
        model(image, prior if model.use_snn_prior else None)

    with torch.inference_mode():
        for _ in range(args.warmup):
            run_once()
        if device.type == "cuda":
            torch.cuda.synchronize()

        latencies_ms: List[float] = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            run_once()
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "image": str(args.image.resolve()),
        "device": str(device),
        "batch_size": 1,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "scope": "model_forward_only_preprocessed_input",
        "mean_ms": statistics.fmean(latencies_ms),
        "p50_ms": percentile(latencies_ms, 0.50),
        "p95_ms": percentile(latencies_ms, 0.95),
        "p99_ms": percentile(latencies_ms, 0.99),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
