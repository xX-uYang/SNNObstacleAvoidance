from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List


CLASS_NAMES = ("left", "straight", "right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inject label-derived fake SNN probabilities for interface debugging. "
            "The resulting numbers must never be reported as model performance."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--simulated-accuracy",
        type=float,
        default=0.72,
        help="Probability that the fake SNN's largest score matches the label.",
    )
    parser.add_argument(
        "--missing-rate",
        type=float,
        default=0.10,
        help="Fraction left blank to exercise the ANN-only fallback path.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def fake_probabilities(
    true_label: int, accuracy: float, rng: random.Random
) -> List[float]:
    if rng.random() < accuracy:
        predicted = true_label
    else:
        predicted = rng.choice(
            [index for index in range(len(CLASS_NAMES)) if index != true_label]
        )
    confidence = rng.uniform(0.55, 0.90)
    residual = 1.0 - confidence
    fraction = rng.uniform(0.20, 0.80)
    other = [index for index in range(len(CLASS_NAMES)) if index != predicted]
    probabilities = [0.0, 0.0, 0.0]
    probabilities[predicted] = confidence
    probabilities[other[0]] = residual * fraction
    probabilities[other[1]] = residual * (1.0 - fraction)
    return probabilities


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.simulated_accuracy <= 1.0:
        raise ValueError("--simulated-accuracy must be in [0, 1].")
    if not 0.0 <= args.missing_rate < 1.0:
        raise ValueError("--missing-rate must be in [0, 1).")

    with args.manifest.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        rows: List[Dict[str, str]] = list(reader)
        fieldnames = list(reader.fieldnames or [])
    required = {"image_path", "label", "split"}
    missing = required - set(fieldnames)
    if missing:
        raise ValueError("Manifest is missing columns: {}".format(sorted(missing)))
    for name in (
        "snn_left",
        "snn_straight",
        "snn_right",
        "snn_source",
    ):
        if name not in fieldnames:
            fieldnames.append(name)

    rng = random.Random(args.seed)
    available = 0
    unavailable = 0
    for row in rows:
        if rng.random() < args.missing_rate:
            row["snn_left"] = ""
            row["snn_straight"] = ""
            row["snn_right"] = ""
            row["snn_source"] = "fake_debug_only_missing"
            unavailable += 1
            continue
        label = int(row["label"])
        if label not in range(len(CLASS_NAMES)):
            raise ValueError("Invalid class label: {}".format(label))
        probabilities = fake_probabilities(
            label, args.simulated_accuracy, rng
        )
        row["snn_left"] = "{:.8f}".format(probabilities[0])
        row["snn_straight"] = "{:.8f}".format(probabilities[1])
        row["snn_right"] = "{:.8f}".format(probabilities[2])
        row["snn_source"] = "fake_debug_only_ground_truth_derived"
        available += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(
        "Created {} fake priors at {}; available={}, missing={}".format(
            len(rows), args.output.resolve(), available, unavailable
        )
    )
    print(
        "WARNING: priors were derived from labels and are for plumbing tests only."
    )


if __name__ == "__main__":
    main()
