from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class SNNPrior:
    """Normalized three-class prior produced by the SNN.

    `available=False` means the ANN should behave as an image-only model.  The
    numeric values are then kept neutral at one third per class.
    """

    left_probability: float = 1.0 / 3.0
    straight_probability: float = 1.0 / 3.0
    right_probability: float = 1.0 / 3.0
    available: bool = False

    @classmethod
    def from_probabilities(
        cls,
        left: Optional[float],
        straight: Optional[float],
        right: Optional[float],
    ) -> "SNNPrior":
        if left is None or straight is None or right is None:
            return cls()
        if left < 0.0 or straight < 0.0 or right < 0.0:
            raise ValueError("SNN probabilities must be non-negative.")
        total = float(left + straight + right)
        if total <= 0.0:
            raise ValueError("At least one SNN probability must be positive.")
        return cls(
            left_probability=float(left) / total,
            straight_probability=float(straight) / total,
            right_probability=float(right) / total,
            available=True,
        )

    @classmethod
    def from_label(cls, label: str) -> "SNNPrior":
        normalized = label.strip().lower()
        if normalized == "left":
            return cls(1.0, 0.0, 0.0, True)
        if normalized == "straight":
            return cls(0.0, 1.0, 0.0, True)
        if normalized == "right":
            return cls(0.0, 0.0, 1.0, True)
        raise ValueError("SNN label must be left, straight, or right.")

    def swapped(self) -> "SNNPrior":
        return SNNPrior(
            left_probability=self.right_probability,
            straight_probability=self.straight_probability,
            right_probability=self.left_probability,
            available=self.available,
        )

    def as_model_input(self) -> Tuple[float, float, float, float]:
        return (
            self.left_probability,
            self.straight_probability,
            self.right_probability,
            1.0 if self.available else 0.0,
        )
