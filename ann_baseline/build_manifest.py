from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path
from typing import Dict, List


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_TO_LABEL = {"left": 0, "straight": 1, "right": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an episode-level train/val/test manifest. Expected layout: "
            "<root>/<episode_id>/{left,straight,right}/*."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def assign_splits(
    episode_names: List[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, str]:
    if len(episode_names) < 3:
        raise ValueError(
            "At least three episodes are required so train, val, and test are "
            "independent. Do not split consecutive frames randomly."
        )
    if train_ratio <= 0.0 or val_ratio <= 0.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("Ratios must be positive and leave room for a test split.")

    shuffled = sorted(episode_names)
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    train_count = max(1, int(count * train_ratio))
    val_count = max(1, int(count * val_ratio))
    if train_count + val_count >= count:
        train_count = count - 2
        val_count = 1

    assignments = {}
    for index, episode in enumerate(shuffled):
        if index < train_count:
            assignments[episode] = "train"
        elif index < train_count + val_count:
            assignments[episode] = "val"
        else:
            assignments[episode] = "test"
    return assignments


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    episodes = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and all((path / class_name).is_dir() for class_name in CLASS_TO_LABEL)
    ]
    if not episodes:
        raise ValueError(
            "No episodes found. Expected left/straight/right below each episode."
        )
    assignments = assign_splits(
        [episode.name for episode in episodes],
        args.train_ratio,
        args.val_ratio,
        args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for episode in sorted(episodes):
        for class_name, label in CLASS_TO_LABEL.items():
            for image_path in sorted((episode / class_name).rglob("*")):
                if (
                    image_path.is_file()
                    and image_path.suffix.lower() in IMAGE_EXTENSIONS
                ):
                    try:
                        stored_path = os.path.relpath(
                            image_path.resolve(), args.output.parent.resolve()
                        )
                    except ValueError:
                        # Windows cannot form a relative path across drive letters.
                        stored_path = str(image_path.resolve())
                    rows.append(
                        {
                            "image_path": stored_path,
                            "split": assignments[episode.name],
                            "label": label,
                            "episode_id": episode.name,
                            "snn_left": "",
                            "snn_straight": "",
                            "snn_right": "",
                            "steering": "",
                        }
                    )
    if not rows:
        raise ValueError("No supported image files were found.")

    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    counts = {"train": 0, "val": 0, "test": 0}
    for row in rows:
        counts[row["split"]] += 1
    print(
        "Created {} with {} rows: train={}, val={}, test={}".format(
            args.output.resolve(),
            len(rows),
            counts["train"],
            counts["val"],
            counts["test"],
        )
    )


if __name__ == "__main__":
    main()
