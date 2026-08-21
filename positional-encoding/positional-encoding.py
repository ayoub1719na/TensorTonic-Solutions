import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    PE = np.zeros((seq_len, d_model))

    positions = np.arange(seq_len)[:, np.newaxis]

    # Number of complete sin/cos pairs
    n_pairs = d_model // 2

    # Frequencies for dimensions 0,2,4,...
    div_term = base ** (2 * np.arange(n_pairs) / d_model)

    # Even dimensions: 0,2,4,...
    PE[:, 0:2 * n_pairs:2] = np.sin(positions / div_term)

    # Odd dimensions: 1,3,5,...
    PE[:, 1:2 * n_pairs:2] = np.cos(positions / div_term)

    # If d_model is odd, last column is sin
    if d_model % 2 == 1:
        last_div_term = base ** (2 * n_pairs / d_model)
        PE[:, -1] = np.sin(positions[:, 0] / last_div_term)

    return PE