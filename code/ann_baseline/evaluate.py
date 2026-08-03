from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from ann_baseline.checkpoint import load_checkpoint
from ann_baseline.constants import CLASS_NAMES
from ann_baseline.data import DirectionDataset
from ann_baseline.metrics import classification_metrics, regression_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved ANN checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
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
    dataset = DirectionDataset(
        manifest_path=args.manifest,
        split=args.split,
        input_size=int(payload["input_size"]),
        training=False,
        task=model.task,
        use_snn_prior=model.use_snn_prior,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    targets: List[float] = []
    predictions: List[float] = []
    records = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            priors = batch["snn_prior"].to(device)
            output = model(images, priors if model.use_snn_prior else None)
            if model.task == "classification":
                probabilities = torch.softmax(output, dim=1).cpu()
                predicted = output.argmax(dim=1).cpu()
                target = batch["label"].cpu()
                targets.extend(target.tolist())
                predictions.extend(predicted.tolist())
                for index, path in enumerate(batch["path"]):
                    records.append(
                        {
                            "image_path": path,
                            "target": CLASS_NAMES[int(target[index])],
                            "prediction": CLASS_NAMES[int(predicted[index])],
                            "prob_left": float(probabilities[index, 0]),
                            "prob_straight": float(probabilities[index, 1]),
                            "prob_right": float(probabilities[index, 2]),
                        }
                    )
            else:
                predicted = output.squeeze(1).cpu()
                target = batch["steering"].cpu()
                targets.extend(target.tolist())
                predictions.extend(predicted.tolist())
                for index, path in enumerate(batch["path"]):
                    records.append(
                        {
                            "image_path": path,
                            "target_steering": float(target[index]),
                            "predicted_steering": float(predicted[index]),
                        }
                    )

    metrics = (
        classification_metrics(
            [int(value) for value in targets],
            [int(value) for value in predictions],
        )
        if model.task == "classification"
        else regression_metrics(targets, predictions)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (args.output_dir / "predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
