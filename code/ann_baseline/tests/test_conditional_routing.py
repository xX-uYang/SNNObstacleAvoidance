"""Tests for interface validation and the three conditional-routing branches."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ann_baseline.conditional_routing import (
    CALL_ANN,
    SAFE_FALLBACK,
    SNN_DIRECT,
    RouterConfig,
    SNNRoutingEvidence,
    SafetyContext,
    decide_route,
)


def valid_snn_row() -> dict[str, object]:
    return {
        "predicted_action": "left",
        "prob_left": 0.96,
        "prob_straight": 0.025,
        "prob_right": 0.015,
        "top1_confidence": 0.96,
        "calibrated_confidence": 0.94,
        "probability_margin": 0.935,
        "membrane_peak_left": 1.25,
        "membrane_peak_straight": 0.65,
        "membrane_peak_right": 0.65,
        "neuron_threshold_left": 1.0,
        "neuron_threshold_straight": 1.0,
        "neuron_threshold_right": 1.0,
        "membrane_ratio_left": 1.25,
        "membrane_ratio_straight": 0.65,
        "membrane_ratio_right": 0.65,
        "any_threshold_crossed": 1,
        "winner_threshold_crossed": 1,
        "temporal_consistency": 0.95,
        "stable_for_steps": 4,
        "input_valid": 1,
        "output_valid": 1,
        "timeout": 0,
        "inference_latency_ms": 8.0,
    }


def valid_safety(**changes: object) -> SafetyContext:
    values = {
        "sensor_valid": True,
        "hazard_present": True,
        "left_lane_available": True,
        "right_lane_available": True,
        "left_lane_safe": True,
        "right_lane_safe": True,
        "speed_mps": 8.0,
        "time_to_collision_s": 1.8,
    }
    values.update(changes)
    return SafetyContext(**values)  # type: ignore[arg-type]


class ConditionalRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RouterConfig()

    def test_high_confidence_safe_direction_exits_at_snn(self) -> None:
        evidence = SNNRoutingEvidence.from_row(valid_snn_row(), self.config)
        decision = decide_route(evidence, valid_safety(), self.config)
        self.assertEqual(decision.route, SNN_DIRECT)

    def test_low_confidence_calls_ann(self) -> None:
        row = valid_snn_row()
        row["calibrated_confidence"] = 0.55
        evidence = SNNRoutingEvidence.from_row(row, self.config)
        decision = decide_route(evidence, valid_safety(), self.config)
        self.assertEqual(decision.route, CALL_ANN)

    def test_unsafe_proposed_lane_calls_ann(self) -> None:
        evidence = SNNRoutingEvidence.from_row(valid_snn_row(), self.config)
        decision = decide_route(
            evidence, valid_safety(left_lane_safe=False), self.config
        )
        self.assertEqual(decision.route, CALL_ANN)

    def test_invalid_sensor_uses_safe_fallback(self) -> None:
        evidence = SNNRoutingEvidence.from_row(valid_snn_row(), self.config)
        decision = decide_route(
            evidence, valid_safety(sensor_valid=False), self.config
        )
        self.assertEqual(decision.route, SAFE_FALLBACK)

    def test_critical_ttc_does_not_wait_for_ann(self) -> None:
        row = valid_snn_row()
        row["calibrated_confidence"] = 0.55
        evidence = SNNRoutingEvidence.from_row(row, self.config)
        decision = decide_route(
            evidence,
            valid_safety(hazard_present=True, time_to_collision_s=0.20),
            self.config,
        )
        self.assertEqual(decision.route, SAFE_FALLBACK)

    def test_probability_sum_error_is_rejected(self) -> None:
        row = valid_snn_row()
        row["prob_right"] = 0.20
        with self.assertRaisesRegex(ValueError, "概率之和"):
            SNNRoutingEvidence.from_row(row, self.config)

    def test_membrane_crossing_flag_mismatch_is_rejected(self) -> None:
        row = valid_snn_row()
        row["winner_threshold_crossed"] = 0
        with self.assertRaisesRegex(ValueError, "winner_threshold_crossed"):
            SNNRoutingEvidence.from_row(row, self.config)


if __name__ == "__main__":
    unittest.main()
