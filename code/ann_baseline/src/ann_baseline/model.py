from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)

from .constants import NUM_CLASSES


class ConditionalANN(nn.Module):
    """Lightweight image model with an optional SNN-prior conditioning branch."""

    def __init__(
        self,
        task: str = "classification",
        use_snn_prior: bool = False,
        pretrained: bool = False,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        if task not in ("classification", "steering"):
            raise ValueError("Task must be 'classification' or 'steering'.")
        self.task = task
        self.use_snn_prior = use_snn_prior
        self.pretrained = pretrained
        self.dropout = float(dropout)

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        image_feature_size = backbone.classifier[0].in_features

        prior_feature_size = 0
        if use_snn_prior:
            prior_feature_size = 32
            self.prior_encoder = nn.Sequential(
                nn.Linear(4, 16),
                nn.Hardswish(),
                nn.Linear(16, prior_feature_size),
                nn.Hardswish(),
            )
        else:
            self.prior_encoder = None

        output_size = NUM_CLASSES if task == "classification" else 1
        self.head = nn.Sequential(
            nn.Linear(image_feature_size + prior_feature_size, 128),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(128, output_size),
        )

    def forward(
        self,
        image: torch.Tensor,
        snn_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        features = self.features(image)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)

        if self.use_snn_prior:
            if snn_prior is None:
                raise ValueError(
                    "This checkpoint requires an SNN prior tensor with shape [B, 4]."
                )
            if snn_prior.ndim != 2 or snn_prior.shape[1] != 4:
                raise ValueError("SNN prior must have shape [batch, 4].")
            prior_features = self.prior_encoder(snn_prior)
            features = torch.cat([features, prior_features], dim=1)

        output = self.head(features)
        if self.task == "steering":
            output = torch.tanh(output)
        return output

    def configuration(self) -> Dict[str, object]:
        return {
            "task": self.task,
            "use_snn_prior": self.use_snn_prior,
            "pretrained": self.pretrained,
            "dropout": self.dropout,
        }


def build_model_from_configuration(
    configuration: Dict[str, object],
    load_pretrained_weights: bool = False,
) -> ConditionalANN:
    return ConditionalANN(
        task=str(configuration["task"]),
        use_snn_prior=bool(configuration["use_snn_prior"]),
        pretrained=(
            bool(configuration.get("pretrained", False))
            if load_pretrained_weights
            else False
        ),
        dropout=float(configuration.get("dropout", 0.20)),
    )
