from typing import List

import numpy as np


def apply_or(operands: List[np.ndarray]) -> np.ndarray:
    return np.min(operands, axis=0)
