"""脚本作用：生成真实图片 + 假接口字段，用于条件路由联调（禁止训练）。

运行顺序中本脚本排第 1。它复用现有 CARLA 前视图片，只模拟还没收到的 SNN
接口字段；输出会醒目标记 fake_debug_only_do_not_train。
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import _fake_routing_data_base as base


_ORIGINAL_TEMPLATES = base.scenario_templates


def checked_scenario_templates():
    """Make every debug template internally consistent before it is written."""

    templates = _ORIGINAL_TEMPLATES()
    names = ("left", "straight", "right")
    for item in templates:
        action = str(item["predicted_action"])
        probabilities = list(item["probabilities"])
        current_winner = probabilities.index(max(probabilities))
        intended_winner = names.index(action)
        if current_winner != intended_winner:
            probabilities[current_winner], probabilities[intended_winner] = (
                probabilities[intended_winner],
                probabilities[current_winner],
            )
            item["probabilities"] = tuple(probabilities)
            item["calibrated_confidence"] = min(
                float(item["calibrated_confidence"]), max(probabilities)
            )
    return templates


base.scenario_templates = checked_scenario_templates


if __name__ == "__main__":
    base.main()
