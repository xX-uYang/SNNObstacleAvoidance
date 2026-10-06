"""Offline conditional-routing rules for the SNN -> ANN prototype.

The module only makes explainable routing decisions.  It never sends a
command to CARLA or to a real vehicle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Tuple


CLASS_NAMES: Tuple[str, str, str] = ("left", "straight", "right")
SNN_DIRECT = "SNN_DIRECT"
CALL_ANN = "CALL_ANN"
SAFE_FALLBACK = "SAFE_FALLBACK"


def parse_bool(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(f"{field_name} 不是可识别的布尔值: {value!r}")


def finite_float(value: object, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是数字，当前为 {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} 必须是有限数字，当前为 {value!r}")
    return number


@dataclass(frozen=True)
class RouterConfig:
    """第一版临时阈值；以后必须用独立验证集重新标定。"""

    min_calibrated_confidence: float = 0.90
    min_probability_margin: float = 0.50
    min_winner_membrane_ratio: float = 1.00
    min_temporal_consistency: float = 0.80
    min_stable_steps: int = 2
    max_snn_latency_ms: float = 100.0
    critical_ttc_s: float = 0.35
    probability_tolerance: float = 1e-4
    membrane_ratio_tolerance: float = 1e-3


@dataclass(frozen=True)
class SafetyContext:
    sensor_valid: bool
    hazard_present: bool
    left_lane_available: bool
    right_lane_available: bool
    left_lane_safe: bool
    right_lane_safe: bool
    speed_mps: float
    time_to_collision_s: float | None

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> "SafetyContext":
        raw_ttc = str(row.get("time_to_collision_s", "") or "").strip()
        ttc = None if not raw_ttc else finite_float(raw_ttc, "time_to_collision_s")
        return cls(
            sensor_valid=parse_bool(row.get("sensor_valid", True), "sensor_valid"),
            hazard_present=parse_bool(
                row.get("hazard_present", False), "hazard_present"
            ),
            left_lane_available=parse_bool(
                row.get("left_lane_available", False), "left_lane_available"
            ),
            right_lane_available=parse_bool(
                row.get("right_lane_available", False), "right_lane_available"
            ),
            left_lane_safe=parse_bool(
                row.get("left_lane_safe", False), "left_lane_safe"
            ),
            right_lane_safe=parse_bool(
                row.get("right_lane_safe", False), "right_lane_safe"
            ),
            speed_mps=finite_float(row.get("speed_mps", 0.0), "speed_mps"),
            time_to_collision_s=ttc,
        )


@dataclass(frozen=True)
class SNNRoutingEvidence:
    probabilities: Tuple[float, float, float]
    membrane_ratios: Tuple[float, float, float]
    predicted_action: str
    top1_confidence: float
    calibrated_confidence: float
    probability_margin: float
    temporal_consistency: float
    stable_for_steps: int
    any_threshold_crossed: bool
    winner_threshold_crossed: bool
    input_valid: bool
    output_valid: bool
    timeout: bool
    inference_latency_ms: float

    @classmethod
    def from_row(
        cls,
        row: Mapping[str, object],
        config: RouterConfig | None = None,
    ) -> "SNNRoutingEvidence":
        cfg = config or RouterConfig()
        probabilities = tuple(
            finite_float(row.get(f"prob_{name}", ""), f"prob_{name}")
            for name in CLASS_NAMES
        )
        if any(probability < 0.0 or probability > 1.0 for probability in probabilities):
            raise ValueError("SNN 三个概率都必须在 0 到 1 之间")
        if abs(sum(probabilities) - 1.0) > cfg.probability_tolerance:
            raise ValueError(f"SNN 三个概率之和必须为 1，当前为 {sum(probabilities):.8f}")

        predicted_action = str(row.get("predicted_action", "")).strip().lower()
        if predicted_action not in CLASS_NAMES:
            raise ValueError("predicted_action 必须是 left、straight 或 right")
        calculated_action = CLASS_NAMES[probabilities.index(max(probabilities))]
        if predicted_action != calculated_action:
            raise ValueError(
                f"predicted_action={predicted_action} 与最大概率类别={calculated_action} 不一致"
            )

        sorted_probabilities = sorted(probabilities, reverse=True)
        calculated_top1 = sorted_probabilities[0]
        calculated_margin = sorted_probabilities[0] - sorted_probabilities[1]
        top1 = finite_float(row.get("top1_confidence", ""), "top1_confidence")
        margin = finite_float(row.get("probability_margin", ""), "probability_margin")
        calibrated = finite_float(
            row.get("calibrated_confidence", ""), "calibrated_confidence"
        )
        if not 0.0 <= calibrated <= 1.0:
            raise ValueError("calibrated_confidence 必须在 0 到 1 之间")
        if abs(top1 - calculated_top1) > cfg.probability_tolerance:
            raise ValueError("top1_confidence 与最大概率不一致")
        if abs(margin - calculated_margin) > cfg.probability_tolerance:
            raise ValueError("probability_margin 与第一名概率减第二名概率不一致")

        ratios = []
        for name in CLASS_NAMES:
            peak = finite_float(
                row.get(f"membrane_peak_{name}", ""), f"membrane_peak_{name}"
            )
            threshold = finite_float(
                row.get(f"neuron_threshold_{name}", ""),
                f"neuron_threshold_{name}",
            )
            if threshold <= 0.0:
                raise ValueError("neuron_threshold 必须大于 0")
            stored_ratio = finite_float(
                row.get(f"membrane_ratio_{name}", ""), f"membrane_ratio_{name}"
            )
            calculated_ratio = peak / threshold
            if abs(stored_ratio - calculated_ratio) > cfg.membrane_ratio_tolerance:
                raise ValueError(f"membrane_ratio_{name} 不等于 peak/threshold")
            ratios.append(stored_ratio)

        any_crossed = parse_bool(
            row.get("any_threshold_crossed", False), "any_threshold_crossed"
        )
        winner_crossed = parse_bool(
            row.get("winner_threshold_crossed", False), "winner_threshold_crossed"
        )
        calculated_any_crossed = max(ratios) >= 1.0
        calculated_winner_crossed = ratios[CLASS_NAMES.index(predicted_action)] >= 1.0
        if any_crossed != calculated_any_crossed:
            raise ValueError("any_threshold_crossed 与膜电位计算结果不一致")
        if winner_crossed != calculated_winner_crossed:
            raise ValueError("winner_threshold_crossed 与获胜类别膜电位计算结果不一致")

        temporal = finite_float(
            row.get("temporal_consistency", ""), "temporal_consistency"
        )
        if not 0.0 <= temporal <= 1.0:
            raise ValueError("temporal_consistency 必须在 0 到 1 之间")
        try:
            stable_steps = int(row.get("stable_for_steps", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("stable_for_steps 必须是整数") from exc
        if stable_steps < 0:
            raise ValueError("stable_for_steps 不能小于 0")
        latency = finite_float(
            row.get("inference_latency_ms", 0.0), "inference_latency_ms"
        )
        if latency < 0.0:
            raise ValueError("inference_latency_ms 不能小于 0")

        return cls(
            probabilities=probabilities,  # type: ignore[arg-type]
            membrane_ratios=tuple(ratios),  # type: ignore[arg-type]
            predicted_action=predicted_action,
            top1_confidence=top1,
            calibrated_confidence=calibrated,
            probability_margin=margin,
            temporal_consistency=temporal,
            stable_for_steps=stable_steps,
            any_threshold_crossed=any_crossed,
            winner_threshold_crossed=winner_crossed,
            input_valid=parse_bool(row.get("input_valid", False), "input_valid"),
            output_valid=parse_bool(row.get("output_valid", False), "output_valid"),
            timeout=parse_bool(row.get("timeout", False), "timeout"),
            inference_latency_ms=latency,
        )

    @property
    def winner_membrane_ratio(self) -> float:
        return self.membrane_ratios[CLASS_NAMES.index(self.predicted_action)]


@dataclass(frozen=True)
class RoutingDecision:
    route: str
    reason: str
    proposed_action: str


def decide_route(
    evidence: SNNRoutingEvidence,
    safety: SafetyContext,
    config: RouterConfig | None = None,
) -> RoutingDecision:
    """Return SNN_DIRECT, CALL_ANN, or SAFE_FALLBACK with a readable reason."""

    cfg = config or RouterConfig()
    action = evidence.predicted_action

    if not safety.sensor_valid:
        return RoutingDecision(
            SAFE_FALLBACK,
            "CARLA 传感器数据无效，不能正常输出方向，进入安全兜底。",
            action,
        )

    def uncertain(reason: str) -> RoutingDecision:
        if (
            safety.hazard_present
            and safety.time_to_collision_s is not None
            and safety.time_to_collision_s <= cfg.critical_ttc_s
        ):
            return RoutingDecision(
                SAFE_FALLBACK,
                reason + "；预计碰撞时间过短，不再等待 ANN，进入安全兜底。",
                action,
            )
        return RoutingDecision(CALL_ANN, reason, action)

    if not evidence.input_valid or not evidence.output_valid:
        return uncertain("SNN 输入或输出被标记为无效，需要 ANN 独立复核。")
    if evidence.timeout or evidence.inference_latency_ms > cfg.max_snn_latency_ms:
        return uncertain("SNN 结果超时或太旧，不能直接采纳。")
    if not evidence.winner_threshold_crossed:
        return uncertain("获胜 SNN 神经元膜电位没有越过阈值。")
    if evidence.winner_membrane_ratio < cfg.min_winner_membrane_ratio:
        return uncertain("获胜膜电位相对阈值仍然不足。")
    if evidence.calibrated_confidence < cfg.min_calibrated_confidence:
        return uncertain("SNN 校准置信度不够。")
    if evidence.probability_margin < cfg.min_probability_margin:
        return uncertain("SNN 第一名和第二名概率太接近，处于灰色地带。")
    if evidence.temporal_consistency < cfg.min_temporal_consistency:
        return uncertain("SNN 最近几个时刻的判断不稳定。")
    if evidence.stable_for_steps < cfg.min_stable_steps:
        return uncertain("SNN 连续保持同一判断的时间还不够。")
    if action == "left" and not (
        safety.left_lane_available and safety.left_lane_safe
    ):
        return uncertain("SNN 建议左转，但左侧道路不可用或不安全。")
    if action == "right" and not (
        safety.right_lane_available and safety.right_lane_safe
    ):
        return uncertain("SNN 建议右转，但右侧道路不可用或不安全。")
    if action == "straight" and safety.hazard_present:
        return uncertain("前方仍有危险，不能直接采纳 SNN 的直行结果。")

    return RoutingDecision(
        SNN_DIRECT,
        "SNN 证据强、连续稳定、没有超时，并且方向通过安全检查。",
        action,
    )


def action_is_safe(action: str, safety: SafetyContext) -> bool:
    """第一版 ANN 输出后的最小安全检查。"""

    if action == "left":
        return safety.left_lane_available and safety.left_lane_safe
    if action == "right":
        return safety.right_lane_available and safety.right_lane_safe
    if action == "straight":
        return not safety.hazard_present
    return False
