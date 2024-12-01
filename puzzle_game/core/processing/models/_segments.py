from dataclasses import dataclass
from ._position import Position
import numpy as np


@dataclass
class Segment:
    """
    Segmento de imagen para el puzzle
    """

    image: np.ndarray
    original_position: Position
    current_position: Position
