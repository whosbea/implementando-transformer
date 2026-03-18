import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Calcula a softmax de forma estável numericamente.
    """
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    """
    Aplica a função ReLU elemento a elemento.
    """
    return np.maximum(0, x)


def layer_norm(x: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Aplica layer normalization no último eixo.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + epsilon)