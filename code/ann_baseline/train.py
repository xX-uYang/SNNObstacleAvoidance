from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from ann_baseline.checkpoint import save_checkpoint
from ann_baseline.constants import CLASS_NAMES, NUM_CLASSES
from ann_baseline.data import DirectionDataset
from ann_baseline.metrics import classification_metrics, regression_metrics
from ann_baseline.model import ConditionalANN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the ANN left/straight/right classifier or steering regressor."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--task",
        choices=("classification", "steering"),
        default="classification",
    )
    parser.add_argument("--use-snn-prior", action="store_true")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Initialize MobileNetV3-Small from ImageNet weights.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-size", type=int, default=160)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--horizontal-flip-probability",
        type=float,
        default=0.0,
        help=(
            "Safe only because the loader also swaps left/right labels and SNN "
            "probabilities, and negates steering."
        ),
    )
    parser.add_argument(
        "--disable-balanced-loss",
        action="store_true",
        help="Disable inverse-frequency class weighting.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def classification_loss(
    dataset: DirectionDataset,
    device: torch.device,
    balanced: bool,
) -> nn.Module:
    if not balanced:
        return nn.CrossEntropyLoss()
    counts = [0 for _ in range(NUM_CLASSES)]
    for row in dataset.rows:
        counts[row.label] += 1
    if min(counts) == 0:
        raise ValueError(
            "Every class is required for training: {}".format(
                ", ".join(CLASS_NAMES)
            )
        )
    total = sum(counts)
    weights = torch.tensor(
        [total / (NUM_CLASSES * count) for count in counts],
        dtype=torch.float32,
        device=device,
    )
    return nn.CrossEntropyLoss(weight=weights)


def run_epoch(
    model: ConditionalANN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW,
    training: bool,
) -> Dict[str, object]:
    model.train(training)
    total_loss = 0.0
    sample_count = 0
    targets: List[float] = []
    predictions: List[float] = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            priors = batch["snn_prior"].to(device, non_blocking=True)
            if model.task == "classification":
                target = batch["label"].to(device, non_blocking=True)
            else:
                target = batch["steering"].to(device, non_blocking=True).unsqueeze(1)

            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(images, priors if model.use_snn_prior else None)
            loss = criterion(output, target)
            if training:
                loss.backward()
                optimizer.step()

            batch_size = images.shape[0]
            total_loss += float(loss.item()) * batch_size
            sample_count += batch_size
            if model.task == "classification":
                targets.extend(target.detach().cpu().tolist())
                predictions.extend(output.argmax(dim=1).detach().cpu().tolist())
            else:
                targets.extend(target.squeeze(1).detach().cpu().tolist())
                predictions.extend(output.squeeze(1).detach().cpu().tolist())

    if model.task == "classification":
        metrics = classification_metrics(
            [int(value) for value in targets],
            [int(value) for value in predictions],
        )
    else:
        metrics = regression_metrics(targets, predictions)
    metrics["loss"] = total_loss / max(sample_count, 1)
    return metrics


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    set_seed(args.seed)
    device = choose_device(args.device)

    train_dataset = DirectionDataset(
        manifest_path=args.manifest,
        split="train",
        input_size=args.input_size,
        training=True,
        task=args.task,
        use_snn_prior=args.use_snn_prior,
        horizontal_flip_probability=args.horizontal_flip_probability,
    )
    val_dataset = DirectionDataset(
        manifest_path=args.manifest,
        split="val",
        input_size=args.input_size,
        training=False,
        task=args.task,
        use_snn_prior=args.use_snn_prior,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = ConditionalANN(
        task=args.task,
        use_snn_prior=args.use_snn_prior,
        pretrained=args.pretrained,
        dropout=args.dropout,
    ).to(device)
    if args.task == "classification":
        criterion = classification_loss(
            train_dataset,
            device,
            balanced=not args.disable_balanced_loss,
        )
    else:
        criterion = nn.SmoothL1Loss(beta=0.1)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max" if args.task == "classification" else "min",
        factor=0.5,
        patience=2,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, object]] = []
    best_value = -float("inf") if args.task == "classification" else float("inf")

    print(
        json.dumps(
            {
                "device": str(device),
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "task": args.task,
                "use_snn_prior": args.use_snn_prior,
            },
            ensure_ascii=False,
        )
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            training=True,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer,
            training=False,
        )
        monitor = (
            float(val_metrics["balanced_accuracy"])
            if args.task == "classification"
            else float(val_metrics["mae"])
        )
        scheduler.step(monitor)

        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

        improved = (
            monitor > best_value
            if args.task == "classification"
            else monitor < best_value
        )
        if improved:
            best_value = monitor
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                epoch,
                args.input_size,
                val_metrics,
            )
        save_checkpoint(
            args.output_dir / "last.pt",
            model,
            epoch,
            args.input_size,
            val_metrics,
        )
        (args.output_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "best_checkpoint": str((args.output_dir / "best.pt").resolve()),
                "best_monitor_value": best_value,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
