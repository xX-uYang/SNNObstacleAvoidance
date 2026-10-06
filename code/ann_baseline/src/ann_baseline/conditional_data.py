"""Read and validate aligned CARLA-frame and SNN-interface CSV files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .conditional_routing import RouterConfig, SNNRoutingEvidence, SafetyContext


KEY_FIELDS = ("sample_id", "episode_id", "frame_id")
CARLA_REQUIRED = {
    *KEY_FIELDS,
    "timestamp_ns",
    "image_path_front",
    "sensor_valid",
    "hazard_present",
    "left_lane_available",
    "right_lane_available",
    "left_lane_safe",
    "right_lane_safe",
    "speed_mps",
    "time_to_collision_s",
    "label_source",
}
SNN_REQUIRED = {
    *KEY_FIELDS,
    "source_timestamp_ns",
    "snn_output_timestamp_ns",
    "predicted_action",
    "prob_left",
    "prob_straight",
    "prob_right",
    "top1_confidence",
    "calibrated_confidence",
    "probability_margin",
    "membrane_peak_left",
    "membrane_peak_straight",
    "membrane_peak_right",
    "neuron_threshold_left",
    "neuron_threshold_straight",
    "neuron_threshold_right",
    "membrane_ratio_left",
    "membrane_ratio_straight",
    "membrane_ratio_right",
    "any_threshold_crossed",
    "winner_threshold_crossed",
    "temporal_consistency",
    "stable_for_steps",
    "input_valid",
    "output_valid",
    "timeout",
    "inference_latency_ms",
    "model_version",
    "calibration_version",
}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到 CSV：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV 没有表头：{path}")
        return [dict(row) for row in reader]


def row_key(row: Mapping[str, object]) -> str:
    return "|".join(str(row.get(field, "")).strip() for field in KEY_FIELDS)


@dataclass(frozen=True)
class AlignedFrame:
    key: str
    carla: Dict[str, str]
    snn: Dict[str, str]
    image_path: Path
    safety: SafetyContext
    evidence: SNNRoutingEvidence


def _check_columns(rows: Sequence[Mapping[str, object]], required: set[str], name: str) -> None:
    if not rows:
        raise ValueError(f"{name} 没有数据行")
    missing = sorted(required - set(rows[0].keys()))
    if missing:
        raise ValueError(f"{name} 缺少字段：{', '.join(missing)}")


def _index_unique(rows: Sequence[Dict[str, str]], name: str) -> Dict[str, Dict[str, str]]:
    indexed: Dict[str, Dict[str, str]] = {}
    for line_number, row in enumerate(rows, start=2):
        key = row_key(row)
        if not all(str(row.get(field, "")).strip() for field in KEY_FIELDS):
            raise ValueError(f"{name} 第 {line_number} 行的关联主键不完整")
        if key in indexed:
            raise ValueError(f"{name} 存在重复主键：{key}")
        indexed[key] = row
    return indexed


def validate_and_align(
    carla_csv: Path,
    snn_csv: Path,
    config: RouterConfig | None = None,
) -> List[AlignedFrame]:
    cfg = config or RouterConfig()
    carla_rows = read_csv_rows(carla_csv)
    snn_rows = read_csv_rows(snn_csv)
    _check_columns(carla_rows, CARLA_REQUIRED, "CARLA CSV")
    _check_columns(snn_rows, SNN_REQUIRED, "SNN CSV")
    carla_index = _index_unique(carla_rows, "CARLA CSV")
    snn_index = _index_unique(snn_rows, "SNN CSV")
    if set(carla_index) != set(snn_index):
        missing_snn = sorted(set(carla_index) - set(snn_index))[:3]
        missing_carla = sorted(set(snn_index) - set(carla_index))[:3]
        raise ValueError(
            "两张表的帧没有一一对应。"
            f" 缺 SNN 的示例={missing_snn}；缺 CARLA 的示例={missing_carla}"
        )

    aligned = []
    for key, carla in carla_index.items():
        snn = snn_index[key]
        if str(carla["timestamp_ns"]).strip() != str(snn["source_timestamp_ns"]).strip():
            raise ValueError(f"{key} 的 CARLA 时间戳和 SNN 输入时间戳不一致")
        try:
            source_time = int(snn["source_timestamp_ns"])
            output_time = int(snn["snn_output_timestamp_ns"])
        except ValueError as exc:
            raise ValueError(f"{key} 的时间戳必须是整数纳秒") from exc
        if output_time < source_time:
            raise ValueError(f"{key} 的 SNN 输出时间早于输入时间")
        image_path = Path(carla["image_path_front"])
        if not image_path.is_absolute():
            image_path = (carla_csv.parent / image_path).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"{key} 对应的前视图片不存在：{image_path}")
        try:
            safety = SafetyContext.from_row(carla)
            evidence = SNNRoutingEvidence.from_row(snn, cfg)
        except ValueError as exc:
            raise ValueError(f"{key} 字段校验失败：{exc}") from exc
        aligned.append(
            AlignedFrame(
                key=key,
                carla=carla,
                snn=snn,
                image_path=image_path,
                safety=safety,
                evidence=evidence,
            )
        )
    return aligned
