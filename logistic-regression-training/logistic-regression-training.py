import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(
        z >= 0,
        1 / (1 + np.exp(-z)),
        np.exp(z) / (1 + np.exp(z))
    )


def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """

    # Convertir en tableaux NumPy
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # Nombre d'exemples et nombre de features
    n_samples, n_features = X.shape

    # Initialisation des paramètres
    w = np.zeros(n_features)
    b = 0.0

    # Gradient descent
    for _ in range(steps):

        # 1. Calcul du score
        z = X @ w + b

        # 2. Calcul des probabilités
        p = _sigmoid(z)

        # 3. Calcul des gradients
        dw = (1 / n_samples) * (X.T @ (p - y))
        db = (1 / n_samples) * np.sum(p - y)

        # 4. Mise à jour des paramètres
        w -= lr * dw
        b -= lr * db

    return w, b