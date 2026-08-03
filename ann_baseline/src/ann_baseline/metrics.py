from __future__ import annotations

import math
from typing import Dict, Iterable, List

from .constants import CLASS_NAMES, NUM_CLASSES


def classification_metrics(
    targets: Iterable[int],
    predictions: Iterable[int],
) -> Dict[str, object]:
    confusion = [
        [0 for _ in range(NUM_CLASSES)] for _ in range(NUM_CLASSES)
    ]
    target_list = list(targets)
    prediction_list = list(predictions)
    if len(target_list) != len(prediction_list):
        raise ValueError("Targets and predictions have different lengths.")
    for target, prediction in zip(target_list, prediction_list):
        confusion[int(target)][int(prediction)] += 1

    total = sum(sum(row) for row in confusion)
    accuracy = (
        sum(confusion[index][index] for index in range(NUM_CLASSES)) / total
        if total
        else float("nan")
    )
    recalls: List[float] = []
    for class_index in range(NUM_CLASSES):
        denominator = sum(confusion[class_index])
        recalls.append(
            confusion[class_index][class_index] / denominator
            if denominator
            else float("nan")
        )
    finite_recalls = [value for value in recalls if not math.isnan(value)]
    balanced_accuracy = (
        sum(finite_recalls) / len(finite_recalls)
        if finite_recalls
        else float("nan")
    )
    result = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "confusion_matrix": confusion,
        "class_order": list(CLASS_NAMES),
        "samples": total,
    }
    for class_name, recall in zip(CLASS_NAMES, recalls):
        result["recall_{}".format(class_name)] = recall
    return result


def regression_metrics(
    targets: Iterable[float],
    predictions: Iterable[float],
) -> Dict[str, float]:
    target_list = list(targets)
    prediction_list = list(predictions)
    if len(target_list) != len(prediction_list):
        raise ValueError("Targets and predictions have different lengths.")
    if not target_list:
        return {"mae": float("nan"), "rmse": float("nan"), "samples": 0}
    errors = [
        float(prediction) - float(target)
        for target, prediction in zip(target_list, prediction_list)
    ]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    return {"mae": mae, "rmse": rmse, "samples": len(errors)}
