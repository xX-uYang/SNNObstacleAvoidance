from __future__ import annotations

import argparse
import csv
import os
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence


CLASS_TO_LABEL = {"left": 0, "straight": 1, "right": 2}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TIMESTAMP_PATTERN = re.compile(
    r"(?P<date>\d{8})_(?P<time>\d{6})_(?P<microseconds>\d{6})"
)
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the current flat CARLA dataset while keeping temporally "
            "adjacent images in the same split."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--new-episode-gap-seconds",
        type=float,
        default=20.0,
        help="Start a new temporal group after this capture-time gap.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def parse_timestamp(path: Path) -> datetime:
    match = TIMESTAMP_PATTERN.search(path.name)
    if match is None:
        raise ValueError(
            "Cannot parse a capture timestamp from filename: {}".format(path)
        )
    return datetime.strptime(
        "{}{}{}".format(
            match.group("date"),
            match.group("time"),
            match.group("microseconds"),
        ),
        "%Y%m%d%H%M%S%f",
    )


def collect_images(root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError("Missing class directory: {}".format(class_dir))
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append(
                    {
                        "image_path": path.resolve(),
                        "class_name": class_name,
                        "label": label,
                        "timestamp": parse_timestamp(path),
                    }
                )
    if not rows:
        raise ValueError("No supported images found under {}".format(root))
    rows.sort(key=lambda row: row["timestamp"])
    return rows


def create_temporal_groups(
    rows: Sequence[Dict[str, object]], gap_seconds: float
) -> List[List[Dict[str, object]]]:
    if gap_seconds <= 0.0:
        raise ValueError("--new-episode-gap-seconds must be positive.")
    groups: List[List[Dict[str, object]]] = []
    current: List[Dict[str, object]] = []
    previous_time = None
    for row in rows:
        timestamp = row["timestamp"]
        if (
            previous_time is not None
            and (timestamp - previous_time).total_seconds() > gap_seconds
        ):
            groups.append(current)
            current = []
        current.append(row)
        previous_time = timestamp
    if current:
        groups.append(current)
    return groups


def assign_groups(
    groups: Sequence[List[Dict[str, object]]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[int, str]:
    test_ratio = 1.0 - train_ratio - val_ratio
    if min(train_ratio, val_ratio, test_ratio) <= 0.0:
        raise ValueError("Split ratios must all be positive.")

    totals = Counter(
        str(row["class_name"]) for group in groups for row in group
    )
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    targets = {
        split: {
            class_name: totals[class_name] * ratio
            for class_name in CLASS_TO_LABEL
        }
        for split, ratio in ratios.items()
    }
    rng = random.Random(seed)
    group_counts = [
        Counter(str(row["class_name"]) for row in group) for group in groups
    ]
    cumulative = [train_ratio, train_ratio + val_ratio]
    best_score = float("inf")
    best_assignments: Dict[int, str] = {}

    # Random search is robust here because there are many temporal groups but only
    # six straight-driving groups. It preserves every group and chooses the most
    # class-balanced split among deterministic seeded candidates.
    for _ in range(20000):
        counts = {split: Counter() for split in SPLITS}
        candidate: Dict[int, str] = {}
        for index, counts_for_group in enumerate(group_counts):
            draw = rng.random()
            split = (
                "train"
                if draw < cumulative[0]
                else "val"
                if draw < cumulative[1]
                else "test"
            )
            candidate[index] = split
            counts[split].update(counts_for_group)

        if any(
            counts[split][class_name] == 0
            for split in SPLITS
            for class_name in CLASS_TO_LABEL
        ):
            continue
        score = 0.0
        for split in SPLITS:
            for class_name in CLASS_TO_LABEL:
                target = max(targets[split][class_name], 1.0)
                score += (
                    (counts[split][class_name] - target) / target
                ) ** 2
        if score < best_score:
            best_score = score
            best_assignments = candidate

    if not best_assignments:
        raise ValueError(
            "Could not create class-complete grouped splits. Try a smaller "
            "--new-episode-gap-seconds."
        )
    return best_assignments


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    rows = collect_images(root)
    groups = create_temporal_groups(rows, args.new_episode_gap_seconds)
    assignments = assign_groups(
        groups, args.train_ratio, args.val_ratio, args.seed
    )

    output_rows = []
    for group_index, group in enumerate(groups):
        episode_id = "time_group_{:03d}".format(group_index + 1)
        for row in group:
            try:
                stored_path = os.path.relpath(
                    row["image_path"], args.output.parent.resolve()
                )
            except ValueError:
                stored_path = str(row["image_path"])
            output_rows.append(
                {
                    "image_path": stored_path,
                    "split": assignments[group_index],
                    "label": row["label"],
                    "class_name": row["class_name"],
                    "episode_id": episode_id,
                    "capture_timestamp": row["timestamp"].isoformat(),
                    "snn_left": "",
                    "snn_straight": "",
                    "snn_right": "",
                    "snn_source": "",
                    "steering": "",
                }
            )
    output_rows.sort(
        key=lambda row: (row["split"], row["capture_timestamp"])
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    counts = Counter(
        (str(row["split"]), str(row["class_name"])) for row in output_rows
    )
    print(
        "Created {} rows in {} temporal groups at {}".format(
            len(output_rows), len(groups), args.output.resolve()
        )
    )
    for split in SPLITS:
        print(
            "{}: {}".format(
                split,
                ", ".join(
                    "{}={}".format(class_name, counts[(split, class_name)])
                    for class_name in CLASS_TO_LABEL
                ),
            )
        )


if __name__ == "__main__":
    main()
