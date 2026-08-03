from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

from .constants import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES
from .interfaces import SNNPrior


@dataclass(frozen=True)
class ManifestRow:
    image_path: Path
    split: str
    label: int
    episode_id: str
    snn_prior: SNNPrior
    steering: float


def parse_label(value: str) -> int:
    normalized = value.strip().lower()
    if normalized == "":
        return -1
    if normalized in ("0", "left"):
        return 0
    if normalized in ("1", "straight"):
        return 1
    if normalized in ("2", "right"):
        return 2
    raise ValueError(
        "Label must be one of: left, straight, right, 0, 1, 2; got {!r}.".format(
            value
        )
    )


def _optional_float(value: Optional[str]) -> Optional[float]:
    if value is None or value.strip() == "":
        return None
    return float(value)


def load_manifest(manifest_path: Path) -> List[ManifestRow]:
    manifest_path = manifest_path.resolve()
    rows: List[ManifestRow] = []
    seen_paths = set()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Manifest is missing columns: {}".format(", ".join(sorted(missing)))
            )
        for line_number, raw in enumerate(reader, start=2):
            raw_path = Path(raw["image_path"])
            image_path = (
                raw_path
                if raw_path.is_absolute()
                else (manifest_path.parent / raw_path).resolve()
            )
            if image_path in seen_paths:
                raise ValueError(
                    "Duplicate image at manifest line {}: {}".format(
                        line_number, image_path
                    )
                )
            if not image_path.is_file():
                raise FileNotFoundError(
                    "Image at manifest line {} does not exist: {}".format(
                        line_number, image_path
                    )
                )
            seen_paths.add(image_path)

            split = raw["split"].strip().lower()
            if split not in ("train", "val", "test"):
                raise ValueError(
                    "Split at line {} must be train, val, or test.".format(
                        line_number
                    )
                )

            snn_left = _optional_float(raw.get("snn_left"))
            snn_straight = _optional_float(raw.get("snn_straight"))
            snn_right = _optional_float(raw.get("snn_right"))
            prior = SNNPrior.from_probabilities(
                snn_left, snn_straight, snn_right
            )
            steering = _optional_float(raw.get("steering"))
            rows.append(
                ManifestRow(
                    image_path=image_path,
                    split=split,
                    label=parse_label(raw.get("label", "")),
                    episode_id=(raw.get("episode_id") or "").strip(),
                    snn_prior=prior,
                    steering=float("nan") if steering is None else float(steering),
                )
            )
    if not rows:
        raise ValueError("Manifest contains no data rows.")
    return rows


def build_transform(input_size: int, training: bool) -> transforms.Compose:
    operations = [transforms.Resize((input_size, input_size))]
    if training:
        operations.extend(
            [
                transforms.ColorJitter(
                    brightness=0.25,
                    contrast=0.25,
                    saturation=0.20,
                    hue=0.03,
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))],
                    p=0.10,
                ),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(operations)


class DirectionDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        split: str,
        input_size: int = 160,
        training: bool = False,
        task: str = "classification",
        use_snn_prior: bool = False,
        horizontal_flip_probability: float = 0.0,
    ) -> None:
        self.rows = [
            row for row in load_manifest(manifest_path) if row.split == split
        ]
        if not self.rows:
            raise ValueError("Manifest has no rows for split {!r}.".format(split))
        if task not in ("classification", "steering"):
            raise ValueError("Task must be 'classification' or 'steering'.")
        if not 0.0 <= horizontal_flip_probability <= 1.0:
            raise ValueError("Horizontal flip probability must be in [0, 1].")
        if task == "steering":
            missing = [row.image_path for row in self.rows if math.isnan(row.steering)]
            if missing:
                raise ValueError(
                    "Steering task requires a steering value for every row; "
                    "{} rows are missing it.".format(len(missing))
                )
        else:
            unlabeled = [
                row.image_path
                for row in self.rows
                if row.label not in range(NUM_CLASSES)
            ]
            if unlabeled:
                raise ValueError(
                    "Classification requires left/straight/right labels; {} rows are "
                    "unlabeled.".format(len(unlabeled))
                )
        if use_snn_prior:
            available = sum(row.snn_prior.available for row in self.rows)
            if available == 0:
                raise ValueError(
                    "SNN fusion was requested, but this split contains no "
                    "snn_left/snn_straight/snn_right values."
                )

        self.transform = build_transform(input_size, training=training)
        self.training = training
        self.task = task
        self.use_snn_prior = use_snn_prior
        self.horizontal_flip_probability = horizontal_flip_probability

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]
        image = Image.open(row.image_path).convert("RGB")
        label = row.label
        prior = row.snn_prior
        steering = row.steering

        should_flip = (
            self.training
            and self.horizontal_flip_probability > 0.0
            and random.random() < self.horizontal_flip_probability
        )
        if should_flip:
            image = ImageOps.mirror(image)
            if label in range(NUM_CLASSES):
                label = {0: 2, 1: 1, 2: 0}[label]
            prior = prior.swapped()
            if not math.isnan(steering):
                steering = -steering

        return {
            "image": self.transform(image),
            "label": torch.tensor(label, dtype=torch.long),
            "snn_prior": torch.tensor(
                prior.as_model_input(), dtype=torch.float32
            ),
            "steering": torch.tensor(steering, dtype=torch.float32),
            "path": str(row.image_path),
            "episode_id": row.episode_id,
            "class_name": (
                CLASS_NAMES[label]
                if label in range(NUM_CLASSES)
                else "unlabeled"
            ),
        }
