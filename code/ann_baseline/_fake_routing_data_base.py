"""脚本作用：用真实 CARLA 图片生成仅供接口联调的假 CARLA 表和假 SNN 输出表。

这些假字段覆盖高置信度直出、低置信度调用 ANN、超时、膜电位未过阈值、
传感器失效和极短 TTC 等情况。它们绝不能用于训练，也不能用于报告准确率。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成条件路由联调用假数据（不训练）")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=WORKSPACE_ROOT / "carla_data" / "carla_data8.6",
        help="包含各 episode/control.csv 的 CARLA 数据根目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE_ROOT / "runs" / "conditional_routing_debug_v1" / "data",
    )
    parser.add_argument("--num-samples", type=int, default=36)
    return parser.parse_args()


def read_source_rows(dataset_root: Path) -> List[Tuple[Dict[str, str], Path]]:
    controls = sorted(dataset_root.glob("episode_*/control.csv"))
    if not controls:
        raise FileNotFoundError(f"没有在 {dataset_root} 下找到 episode_*/control.csv")
    records: List[Tuple[Dict[str, str], Path]] = []
    for control_path in controls:
        with control_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                image_value = row.get("image_path_straight") or row.get("image_path")
                if not image_value:
                    continue
                image_path = Path(image_value)
                if not image_path.is_absolute():
                    image_path = (dataset_root / image_path).resolve()
                if image_path.is_file():
                    copied = dict(row)
                    copied["_source_control_csv"] = str(control_path.resolve())
                    copied["_front_image_absolute"] = str(image_path)
                    records.append((copied, control_path))
    if not records:
        raise ValueError("找到了 control.csv，但没有一行能对应到真实存在的前视图片")
    return records


def evenly_select(
    records: List[Tuple[Dict[str, str], Path]], count: int
) -> List[Tuple[Dict[str, str], Path]]:
    if count <= 0:
        raise ValueError("--num-samples 必须大于 0")
    count = min(count, len(records))
    if count == 1:
        return [records[0]]
    indexes = [round(index * (len(records) - 1) / (count - 1)) for index in range(count)]
    return [records[index] for index in indexes]


def scenario_templates() -> List[Dict[str, object]]:
    base = {
        "sensor_valid": True,
        "hazard_present": True,
        "left_lane_available": True,
        "right_lane_available": True,
        "left_lane_safe": True,
        "right_lane_safe": True,
        "time_to_collision_s": 1.8,
        "input_valid": True,
        "output_valid": True,
        "timeout": False,
        "inference_latency_ms": 8.0,
        "temporal_consistency": 0.95,
        "stable_for_steps": 4,
    }

    def make(name: str, action: str, probabilities: Tuple[float, float, float], **changes: object) -> Dict[str, object]:
        item = dict(base)
        item.update(
            {
                "debug_scenario": name,
                "predicted_action": action,
                "probabilities": probabilities,
                "calibrated_confidence": max(probabilities),
                "membrane_ratios": tuple(
                    1.25 if class_name == action else 0.65
                    for class_name in ("left", "straight", "right")
                ),
            }
        )
        item.update(changes)
        return item

    return [
        make("高置信度左转且左侧安全", "left", (0.96, 0.025, 0.015)),
        make("高置信度右转且右侧安全", "right", (0.015, 0.025, 0.96)),
        make(
            "高置信度直行且前方无危险",
            "straight",
            (0.02, 0.96, 0.02),
            hazard_present=False,
            time_to_collision_s="",
        ),
        make("置信度低", "left", (0.45, 0.32, 0.23), calibrated_confidence=0.42),
        make("第一名和第二名太接近", "left", (0.48, 0.47, 0.05)),
        make(
            "获胜膜电位未过阈值",
            "right",
            (0.96, 0.02, 0.02),
            membrane_ratios=(0.60, 0.55, 0.95),
        ),
        make(
            "连续帧判断不稳定",
            "left",
            (0.96, 0.025, 0.015),
            temporal_consistency=0.45,
            stable_for_steps=1,
        ),
        make(
            "左侧不安全",
            "left",
            (0.96, 0.025, 0.015),
            left_lane_safe=False,
        ),
        make(
            "SNN超时",
            "right",
            (0.015, 0.025, 0.96),
            timeout=True,
            inference_latency_ms=140.0,
        ),
        make(
            "SNN输出无效",
            "left",
            (0.96, 0.025, 0.015),
            output_valid=False,
        ),
        make(
            "极紧急但SNN不确定",
            "left",
            (0.45, 0.32, 0.23),
            calibrated_confidence=0.42,
            time_to_collision_s=0.20,
        ),
        make(
            "CARLA传感器无效",
            "right",
            (0.015, 0.025, 0.96),
            sensor_valid=False,
        ),
    ]


def resolve_optional_image(dataset_root: Path, value: str) -> str:
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        path = (dataset_root / path).resolve()
    return str(path) if path.is_file() else ""


def build_rows(
    selected: Iterable[Tuple[Dict[str, str], Path]], dataset_root: Path
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    templates = scenario_templates()
    carla_rows: List[Dict[str, object]] = []
    snn_rows: List[Dict[str, object]] = []
    for index, (source, _) in enumerate(selected):
        template = templates[index % len(templates)]
        frame_id = source.get("frame_id") or str(index)
        episode_id = source.get("episode_id") or "unknown_episode"
        sample_id = source.get("sample_id") or f"{episode_id}_{frame_id}"
        raw_timestamp = source.get("timestamp_ns", "")
        if raw_timestamp:
            timestamp_ns = int(float(raw_timestamp))
        else:
            timestamp_ns = int(float(source.get("simulation_time_s", index)) * 1e9)
        latency_ms = float(template["inference_latency_ms"])
        output_timestamp_ns = timestamp_ns + round(latency_ms * 1e6)
        probabilities = tuple(template["probabilities"])  # type: ignore[arg-type]
        sorted_probabilities = sorted(probabilities, reverse=True)
        ratios = tuple(template["membrane_ratios"])  # type: ignore[arg-type]
        action = str(template["predicted_action"])
        winner_index = ("left", "straight", "right").index(action)

        carla_rows.append(
            {
                "schema_version": "carla_conditional_debug_v1",
                "sample_id": sample_id,
                "episode_id": episode_id,
                "frame_id": frame_id,
                "timestamp_ns": timestamp_ns,
                "simulation_time_s": source.get("simulation_time_s", ""),
                "image_path_front": source["_front_image_absolute"],
                "image_path_left": resolve_optional_image(
                    dataset_root, source.get("image_path_left", "")
                ),
                "image_path_right": resolve_optional_image(
                    dataset_root, source.get("image_path_right", "")
                ),
                "camera_frame_id_front": frame_id,
                "max_sync_delta_ms": 0.0,
                "sensor_valid": int(bool(template["sensor_valid"])),
                "speed_mps": source.get("speed_mps") or source.get("speed") or 0.0,
                "steer_applied": source.get("steer", ""),
                "throttle_applied": source.get("throttle", ""),
                "brake_applied": source.get("brake", ""),
                "road_option": source.get("road_option", ""),
                "hazard_present": int(bool(template["hazard_present"])),
                "left_lane_available": int(bool(template["left_lane_available"])),
                "right_lane_available": int(bool(template["right_lane_available"])),
                "left_lane_safe": int(bool(template["left_lane_safe"])),
                "right_lane_safe": int(bool(template["right_lane_safe"])),
                "time_to_collision_s": template["time_to_collision_s"],
                "debug_scenario": template["debug_scenario"],
                "source_true_label_for_reference_only": source.get("true_label", ""),
                "label_source": "fake_debug_only_do_not_train",
                "source_control_csv": source["_source_control_csv"],
            }
        )

        threshold = 1.0
        snn_rows.append(
            {
                "schema_version": "snn_ann_interface_debug_v1",
                "sample_id": sample_id,
                "episode_id": episode_id,
                "frame_id": frame_id,
                "source_timestamp_ns": timestamp_ns,
                "snn_output_timestamp_ns": output_timestamp_ns,
                "predicted_action": action,
                "prob_left": probabilities[0],
                "prob_straight": probabilities[1],
                "prob_right": probabilities[2],
                "top1_confidence": sorted_probabilities[0],
                "calibrated_confidence": template["calibrated_confidence"],
                "probability_margin": sorted_probabilities[0] - sorted_probabilities[1],
                "membrane_peak_left": ratios[0] * threshold,
                "membrane_peak_straight": ratios[1] * threshold,
                "membrane_peak_right": ratios[2] * threshold,
                "neuron_threshold_left": threshold,
                "neuron_threshold_straight": threshold,
                "neuron_threshold_right": threshold,
                "membrane_ratio_left": ratios[0],
                "membrane_ratio_straight": ratios[1],
                "membrane_ratio_right": ratios[2],
                "any_threshold_crossed": int(max(ratios) >= 1.0),
                "winner_threshold_crossed": int(ratios[winner_index] >= 1.0),
                "spike_count_left": 8 if action == "left" and max(ratios) >= 1.0 else 1,
                "spike_count_straight": 8 if action == "straight" and max(ratios) >= 1.0 else 1,
                "spike_count_right": 8 if action == "right" and max(ratios) >= 1.0 else 1,
                "first_spike_time_step": 3 if max(ratios) >= 1.0 else "",
                "temporal_consistency": template["temporal_consistency"],
                "stable_for_steps": template["stable_for_steps"],
                "input_valid": int(bool(template["input_valid"])),
                "output_valid": int(bool(template["output_valid"])),
                "timeout": int(bool(template["timeout"])),
                "inference_latency_ms": latency_ms,
                "model_version": "fake_snn_debug_v1_not_a_real_model",
                "calibration_version": "fake_identity_mapping_v0",
                "debug_scenario": template["debug_scenario"],
            }
        )
    return carla_rows, snn_rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    records = read_source_rows(dataset_root)
    selected = evenly_select(records, args.num_samples)
    carla_rows, snn_rows = build_rows(selected, dataset_root)
    carla_path = output_dir / "carla_frames_fake_v1.csv"
    snn_path = output_dir / "snn_output_fake_v1.csv"
    write_csv(carla_path, carla_rows)
    write_csv(snn_path, snn_rows)
    metadata = {
        "purpose": "interface_and_routing_debug_only",
        "warning": "fake fields; never use for training or accuracy reporting",
        "dataset_root": str(dataset_root),
        "sample_count": len(carla_rows),
        "carla_csv": str(carla_path),
        "snn_csv": str(snn_path),
        "scenario_count": len(scenario_templates()),
    }
    metadata_path = output_dir / "fake_data_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
