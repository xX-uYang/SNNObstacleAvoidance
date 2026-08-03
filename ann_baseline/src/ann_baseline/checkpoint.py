from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch

from .model import ConditionalANN, build_model_from_configuration
from .constants import CLASS_NAMES


def save_checkpoint(
    path: Path,
    model: ConditionalANN,
    epoch: int,
    input_size: int,
    metrics: Dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_configuration": model.configuration(),
            "model_state": model.state_dict(),
            "input_size": int(input_size),
            "metrics": metrics,
            "class_names": list(CLASS_NAMES),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> Tuple[ConditionalANN, Dict[str, object]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_model_from_configuration(payload["model_configuration"])
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, payload
