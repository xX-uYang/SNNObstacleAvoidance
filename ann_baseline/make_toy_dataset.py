from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a tiny synthetic dataset for pipeline smoke tests."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--samples-per-class", type=int, default=16)
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def make_image(
    size: int,
    direction: str,
    rng: random.Random,
) -> Image.Image:
    sky = (rng.randint(65, 105), rng.randint(95, 145), rng.randint(140, 190))
    road = (rng.randint(45, 75),) * 3
    image = Image.new("RGB", (size, size), sky)
    draw = ImageDraw.Draw(image)
    horizon = int(size * 0.30)
    draw.rectangle((0, horizon, size, size), fill=road)

    sign = -1.0 if direction == "left" else 1.0 if direction == "right" else 0.0
    center_points = []
    left_points = []
    right_points = []
    for y in range(size - 1, horizon - 1, -3):
        progress = (size - 1 - y) / max(size - 1 - horizon, 1)
        center = size * 0.5 + sign * (progress ** 1.7) * size * 0.26
        half_width = (1.0 - progress) * size * 0.34 + progress * size * 0.07
        jitter = rng.uniform(-0.8, 0.8)
        center_points.append((center + jitter, y))
        left_points.append((center - half_width + jitter, y))
        right_points.append((center + half_width + jitter, y))

    draw.line(left_points, fill=(235, 235, 220), width=max(2, size // 45))
    draw.line(right_points, fill=(235, 235, 220), width=max(2, size // 45))
    draw.line(
        center_points[::2],
        fill=(245, 196, 45),
        width=max(2, size // 60),
    )

    for _ in range(rng.randint(2, 7)):
        x = rng.randint(0, size - 1)
        y = rng.randint(horizon, size - 1)
        radius = rng.randint(1, max(2, size // 45))
        shade = rng.randint(30, 95)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(shade,) * 3)

    brightness = ImageEnhance.Brightness(image).enhance(rng.uniform(0.75, 1.25))
    contrast = ImageEnhance.Contrast(brightness).enhance(rng.uniform(0.85, 1.20))
    if rng.random() < 0.20:
        contrast = contrast.filter(ImageFilter.GaussianBlur(radius=0.6))
    return contrast


def main() -> None:
    args = parse_args()
    if args.episodes < 3:
        raise ValueError("At least three episodes are required.")
    rng = random.Random(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    total = 0
    for episode_index in range(args.episodes):
        episode_dir = args.output / "episode_{:03d}".format(episode_index + 1)
        for direction in ("left", "straight", "right"):
            class_dir = episode_dir / direction
            class_dir.mkdir(parents=True, exist_ok=True)
            for sample_index in range(args.samples_per_class):
                image = make_image(args.size, direction, rng)
                image.save(
                    class_dir / "{:04d}.png".format(sample_index),
                    compress_level=4,
                )
                total += 1
    print("Created {} toy images under {}".format(total, args.output.resolve()))


if __name__ == "__main__":
    main()
