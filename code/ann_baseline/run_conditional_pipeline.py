"""脚本作用：运行 SNN 条件路由；只有 CALL_ANN 的帧才执行现有 ANN 推理。

本脚本是离线联调程序，不向 CARLA 或真实车辆发送控制指令。当前旧 ANN 仅输出
left/straight/right；若 ANN 给出的方向不通过安全检查，日志写入 brake_fallback，
它只是待后续实现的安全兜底标记，不是已经完成的刹车控制器。
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from PIL import Image

from ann_baseline.checkpoint import load_checkpoint
from ann_baseline.conditional_data import AlignedFrame, validate_and_align
from ann_baseline.conditional_routing import (
    CALL_ANN,
    SAFE_FALLBACK,
    SNN_DIRECT,
    RouterConfig,
    action_is_safe,
    decide_route,
)
from ann_baseline.constants import CLASS_NAMES
from ann_baseline.data import build_transform


def parse_args() -> argparse.Namespace:
    data_dir = WORKSPACE_ROOT / "runs" / "conditional_routing_debug_v1" / "data"
    parser = argparse.ArgumentParser(description="运行条件路由全链路联调（不训练）")
    parser.add_argument(
        "--carla-csv", type=Path, default=data_dir / "carla_frames_fake_v1.csv"
    )
    parser.add_argument(
        "--snn-csv", type=Path, default=data_dir / "snn_output_fake_v1.csv"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=WORKSPACE_ROOT / "runs" / "ann_image_v1" / "best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE_ROOT / "runs" / "conditional_routing_debug_v1" / "result",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--min-snn-confidence", type=float, default=0.90)
    parser.add_argument("--min-probability-margin", type=float, default=0.50)
    parser.add_argument("--critical-ttc-s", type=float, default=0.35)
    return parser.parse_args()


class LazyANN:
    """Load the ANN once, and call it only for frames routed to CALL_ANN."""

    def __init__(self, checkpoint: Path, device_name: str) -> None:
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA，但当前环境检测不到可用 CUDA")
        self.device = torch.device(
            "cuda"
            if device_name == "cuda"
            or (device_name == "auto" and torch.cuda.is_available())
            else "cpu"
        )
        self.checkpoint = checkpoint
        self.model = None
        self.payload = None
        self.transform = None
        self.load_latency_ms = 0.0

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"找不到 ANN checkpoint：{self.checkpoint}")
        started = time.perf_counter()
        model, payload = load_checkpoint(self.checkpoint, self.device)
        if model.task != "classification":
            raise ValueError("当前条件路由调试程序只接分类 ANN checkpoint")
        if model.use_snn_prior:
            raise ValueError(
                "该 checkpoint 会把 SNN 当 ANN 输入；当前选择的是条件路由方案，"
                "请使用 image-only ANN checkpoint"
            )
        self.model = model
        self.payload = payload
        self.transform = build_transform(int(payload["input_size"]), training=False)
        self.load_latency_ms = (time.perf_counter() - started) * 1000.0

    def predict(self, image_path: Path) -> Dict[str, object]:
        self.load()
        assert self.model is not None
        assert self.transform is not None
        image = self.transform(Image.open(image_path).convert("RGB")).unsqueeze(0)
        image = image.to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model(image, None)
            probabilities = torch.softmax(output, dim=1)[0].cpu().tolist()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        predicted_index = int(max(range(len(probabilities)), key=probabilities.__getitem__))
        return {
            "action": CLASS_NAMES[predicted_index],
            "probabilities": probabilities,
            "latency_ms": latency_ms,
        }


def base_log(frame: AlignedFrame) -> Dict[str, object]:
    return {
        "sample_id": frame.carla["sample_id"],
        "episode_id": frame.carla["episode_id"],
        "frame_id": frame.carla["frame_id"],
        "timestamp_ns": frame.carla["timestamp_ns"],
        "image_path_front": str(frame.image_path),
        "debug_scenario": frame.carla.get("debug_scenario", ""),
        "snn_action": frame.evidence.predicted_action,
        "snn_prob_left": frame.evidence.probabilities[0],
        "snn_prob_straight": frame.evidence.probabilities[1],
        "snn_prob_right": frame.evidence.probabilities[2],
        "snn_top1_confidence": frame.evidence.top1_confidence,
        "snn_calibrated_confidence": frame.evidence.calibrated_confidence,
        "snn_probability_margin": frame.evidence.probability_margin,
        "snn_winner_membrane_ratio": frame.evidence.winner_membrane_ratio,
        "snn_temporal_consistency": frame.evidence.temporal_consistency,
        "snn_stable_for_steps": frame.evidence.stable_for_steps,
        "snn_latency_ms": frame.evidence.inference_latency_ms,
        "hazard_present": int(frame.safety.hazard_present),
        "time_to_collision_s": (
            "" if frame.safety.time_to_collision_s is None else frame.safety.time_to_collision_s
        ),
        "left_lane_safe": int(frame.safety.left_lane_safe),
        "right_lane_safe": int(frame.safety.right_lane_safe),
    }


def process_frame(
    frame: AlignedFrame, config: RouterConfig, ann: LazyANN
) -> Dict[str, object]:
    decision = decide_route(frame.evidence, frame.safety, config)
    row = base_log(frame)
    row.update(
        {
            "route": decision.route,
            "route_reason": decision.reason,
            "ann_called": 0,
            "ann_action": "",
            "ann_prob_left": "",
            "ann_prob_straight": "",
            "ann_prob_right": "",
            "ann_latency_ms": 0.0,
            "safety_override": 0,
            "final_action": "",
            "final_action_source": "",
        }
    )
    if decision.route == SNN_DIRECT:
        row["final_action"] = decision.proposed_action
        row["final_action_source"] = "snn_direct"
        return row
    if decision.route == SAFE_FALLBACK:
        row["final_action"] = "brake_fallback"
        row["final_action_source"] = "rule_based_safety_fallback_placeholder"
        row["safety_override"] = 1
        return row
    if decision.route != CALL_ANN:
        raise RuntimeError(f"未知路由：{decision.route}")

    prediction = ann.predict(frame.image_path)
    probabilities = prediction["probabilities"]
    ann_action = str(prediction["action"])
    row.update(
        {
            "ann_called": 1,
            "ann_action": ann_action,
            "ann_prob_left": probabilities[0],  # type: ignore[index]
            "ann_prob_straight": probabilities[1],  # type: ignore[index]
            "ann_prob_right": probabilities[2],  # type: ignore[index]
            "ann_latency_ms": prediction["latency_ms"],
        }
    )
    if action_is_safe(ann_action, frame.safety):
        row["final_action"] = ann_action
        row["final_action_source"] = "ann_after_snn_review"
    else:
        row["final_action"] = "brake_fallback"
        row["final_action_source"] = "rule_based_safety_fallback_placeholder"
        row["safety_override"] = 1
    return row


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = RouterConfig(
        min_calibrated_confidence=args.min_snn_confidence,
        min_probability_margin=args.min_probability_margin,
        critical_ttc_s=args.critical_ttc_s,
    )
    frames = validate_and_align(args.carla_csv.resolve(), args.snn_csv.resolve(), config)
    ann = LazyANN(args.checkpoint.resolve(), args.device)
    started = time.perf_counter()
    rows = [process_frame(frame, config, ann) for frame in frames]
    pipeline_latency_ms = (time.perf_counter() - started) * 1000.0
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "routing_log_v1.csv"
    write_csv(log_path, rows)
    route_counts = Counter(str(row["route"]) for row in rows)
    final_counts = Counter(str(row["final_action"]) for row in rows)
    ann_latencies = [float(row["ann_latency_ms"]) for row in rows if row["ann_called"] == 1]
    summary = {
        "status": "success",
        "warning": "offline fake-interface debug; not a safety validation and not an accuracy result",
        "frame_count": len(rows),
        "route_counts": dict(route_counts),
        "ann_call_count": len(ann_latencies),
        "ann_call_rate": len(ann_latencies) / len(rows),
        "confirmed_no_ann_count": sum(1 for row in rows if row["ann_called"] == 0),
        "final_action_counts": dict(final_counts),
        "safety_override_count": sum(int(row["safety_override"]) for row in rows),
        "ann_device": str(ann.device),
        "ann_checkpoint": str(ann.checkpoint),
        "ann_model_loaded": ann.model is not None,
        "ann_model_load_latency_ms": ann.load_latency_ms,
        "ann_mean_inference_latency_ms": statistics.mean(ann_latencies) if ann_latencies else 0.0,
        "ann_max_inference_latency_ms": max(ann_latencies) if ann_latencies else 0.0,
        "whole_offline_run_latency_ms": pipeline_latency_ms,
        "router_thresholds": {
            "min_calibrated_confidence": config.min_calibrated_confidence,
            "min_probability_margin": config.min_probability_margin,
            "min_winner_membrane_ratio": config.min_winner_membrane_ratio,
            "min_temporal_consistency": config.min_temporal_consistency,
            "min_stable_steps": config.min_stable_steps,
            "max_snn_latency_ms": config.max_snn_latency_ms,
            "critical_ttc_s": config.critical_ttc_s,
        },
        "routing_log": str(log_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
