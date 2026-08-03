from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge Huang Qiyin's per-image SNN probabilities into an ANN manifest. "
            "The SNN CSV must contain image_path,snn_left,snn_straight,snn_right."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--snn-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def normalized_path(raw: str, base: Path) -> str:
    path = Path(raw)
    resolved = path.resolve() if path.is_absolute() else (base / path).resolve()
    return str(resolved).lower()


def main() -> None:
    args = parse_args()
    predictions: Dict[str, Dict[str, str]] = {}
    with args.snn_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "image_path",
            "snn_left",
            "snn_straight",
            "snn_right",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("SNN CSV is missing: {}".format(sorted(missing)))
        for row in reader:
            key = normalized_path(row["image_path"], args.snn_csv.parent)
            if key in predictions:
                raise ValueError("Duplicate SNN prediction for {}".format(key))
            predictions[key] = row

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for name in ("snn_left", "snn_straight", "snn_right"):
            if name not in fieldnames:
                fieldnames.append(name)
        rows = list(reader)

    missing_count = 0
    for row in rows:
        key = normalized_path(row["image_path"], args.manifest.parent)
        prediction = predictions.get(key)
        if prediction is None:
            missing_count += 1
            row["snn_left"] = ""
            row["snn_straight"] = ""
            row["snn_right"] = ""
        else:
            row["snn_left"] = prediction["snn_left"]
            row["snn_straight"] = prediction["snn_straight"]
            row["snn_right"] = prediction["snn_right"]

    if missing_count and not args.allow_missing:
        raise ValueError(
            "{} manifest rows have no SNN output. Re-run with --allow-missing only "
            "for deliberate ablation experiments.".format(missing_count)
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(
        "Merged {} rows into {}; missing={}".format(
            len(rows), args.output.resolve(), missing_count
        )
    )


if __name__ == "__main__":
    main()
