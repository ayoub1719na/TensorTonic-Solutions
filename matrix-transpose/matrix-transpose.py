from typing import Union

import numpy as np

def matrix_transpose(A: Union[list[list[float]], np.ndarray]) -> np.ndarray:
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    return np.array(A).T
    
