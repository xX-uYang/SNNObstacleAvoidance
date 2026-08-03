"""ANN baseline for the SNN+ANN emergency-avoidance project."""

from .constants import CLASS_NAMES, LEFT, NUM_CLASSES, RIGHT, STRAIGHT
from .interfaces import SNNPrior

__all__ = [
    "CLASS_NAMES",
    "LEFT",
    "STRAIGHT",
    "RIGHT",
    "NUM_CLASSES",
    "SNNPrior",
]
