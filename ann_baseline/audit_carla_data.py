from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional

from PIL import Image


CLASS_NAMES = ("left", "straight", "right")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TIMESTAMP_PATTERN = re.compile(
    r"(?P<date>\d{8})_(?P<time>\d{6})_(?P<microseconds>\d{6})"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the flat CARLA left/straight/right image dataset."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_timestamp(name: str) -> Optional[datetime]:
    match = TIMESTAMP_PATTERN.search(name)
    if match is None:
        return None
    text = "{}{}{}".format(
        match.group("date"),
        match.group("time"),
        match.group("microseconds"),
    )
    return datetime.strptime(text, "%Y%m%d%H%M%S%f")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(image: Image.Image, width: int = 9, height: int = 8) -> str:
    grayscale = image.convert("L").resize((width, height))
    pixels = list(grayscale.getdata())
    bits = []
    for y in range(height):
        row = pixels[y * width : (y + 1) * width]
        bits.extend(left > right for left, right in zip(row, row[1:]))
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return "{:016x}".format(value)


def quantiles(values: Iterable[float]) -> Dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {}

    def percentile(p: float) -> float:
        index = (len(ordered) - 1) * p
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = index - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    return {
        "min": ordered[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    records: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []
    for class_name in CLASS_NAMES:
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(
                "Missing class directory: {}".format(class_dir)
            )
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            timestamp = parse_timestamp(path.name)
            try:
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
                    mode = image.mode
                    dhash = difference_hash(image)
                records.append(
                    {
                        "image_path": str(path.resolve()),
                        "class_name": class_name,
                        "width": width,
                        "height": height,
                        "mode": mode,
                        "file_bytes": path.stat().st_size,
                        "timestamp": timestamp.isoformat()
                        if timestamp is not None
                        else "",
                        "sha256": file_sha256(path),
                        "dhash": dhash,
                    }
                )
            except Exception as exc:
                failures.append(
                    {"image_path": str(path.resolve()), "error": repr(exc)}
                )

    if not records:
        raise ValueError("No supported images found under {}".format(root))

    by_class = Counter(str(row["class_name"]) for row in records)
    dimensions = Counter(
        "{}x{}".format(row["width"], row["height"]) for row in records
    )
    modes = Counter(str(row["mode"]) for row in records)
    sha_groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    dhash_groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in records:
        sha_groups[str(row["sha256"])].append(row)
        dhash_groups[str(row["dhash"])].append(row)
    exact_duplicate_groups = [
        group for group in sha_groups.values() if len(group) > 1
    ]
    perceptual_collision_groups = [
        group for group in dhash_groups.values() if len(group) > 1
    ]

    temporal: Dict[str, object] = {}
    for class_name in CLASS_NAMES:
        timed_rows = [
            row
            for row in records
            if row["class_name"] == class_name and row["timestamp"]
        ]
        timed_rows.sort(key=lambda row: str(row["timestamp"]))
        gaps = [
            (
                datetime.fromisoformat(str(current["timestamp"]))
                - datetime.fromisoformat(str(previous["timestamp"]))
            ).total_seconds()
            for previous, current in zip(timed_rows, timed_rows[1:])
        ]
        temporal[class_name] = {
            "parsed_timestamps": len(timed_rows),
            "first": timed_rows[0]["timestamp"] if timed_rows else None,
            "last": timed_rows[-1]["timestamp"] if timed_rows else None,
            "gap_seconds": quantiles(gaps),
            "largest_gaps_seconds": sorted(gaps, reverse=True)[:15],
            "median_gap_seconds": median(gaps) if gaps else None,
        }

    cross_class_exact = []
    for group in exact_duplicate_groups:
        labels = sorted({str(row["class_name"]) for row in group})
        if len(labels) > 1:
            cross_class_exact.append(
                {
                    "classes": labels,
                    "paths": [str(row["image_path"]) for row in group],
                }
            )

    summary = {
        "dataset_root": str(root),
        "total_valid_images": len(records),
        "corrupt_or_unreadable": failures,
        "class_counts": dict(by_class),
        "dimensions": dict(dimensions),
        "modes": dict(modes),
        "file_size_bytes": quantiles(
            float(row["file_bytes"]) for row in records
        ),
        "exact_duplicate_groups": len(exact_duplicate_groups),
        "images_in_exact_duplicate_groups": sum(
            len(group) for group in exact_duplicate_groups
        ),
        "cross_class_exact_duplicate_groups": cross_class_exact,
        "same_dhash_groups": len(perceptual_collision_groups),
        "images_in_same_dhash_groups": sum(
            len(group) for group in perceptual_collision_groups
        ),
        "temporal_profile": temporal,
        "limitations": [
            "Equal dHash is only a coarse near-duplicate signal, not proof of identity.",
            "Filename timestamps are assumed to describe capture order.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "carla_data_audit.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "carla_data_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
