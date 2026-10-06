"""脚本作用：在运行 ANN 前校验 CARLA 表与 SNN 接口表是否完整且逐帧对齐。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ann_baseline.conditional_data import validate_and_align
from ann_baseline.conditional_routing import RouterConfig, decide_route


def parse_args() -> argparse.Namespace:
    data_dir = WORKSPACE_ROOT / "runs" / "conditional_routing_debug_v1" / "data"
    parser = argparse.ArgumentParser(description="校验条件路由输入数据（不训练）")
    parser.add_argument(
        "--carla-csv", type=Path, default=data_dir / "carla_frames_fake_v1.csv"
    )
    parser.add_argument(
        "--snn-csv", type=Path, default=data_dir / "snn_output_fake_v1.csv"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RouterConfig()
    frames = validate_and_align(args.carla_csv.resolve(), args.snn_csv.resolve(), config)
    decisions = [decide_route(frame.evidence, frame.safety, config) for frame in frames]
    report = {
        "validation_passed": True,
        "aligned_frame_count": len(frames),
        "route_preview_counts": dict(Counter(item.route for item in decisions)),
        "message": "字段、概率、膜电位比值、时间戳、图片路径和逐帧对应关系均通过检查。",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
