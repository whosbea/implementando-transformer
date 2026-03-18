import numpy as np
from math_utils import relu


def initialize_ffn_weights(d_model: int, d_ff: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Inicializa os pesos e biases da Feed-Forward Network.

    Shapes:
    - W1: (d_model, d_ff)
    - b1: (d_ff,)
    - W2: (d_ff, d_model)
    - b2: (d_model,)
    """
    w1 = np.random.randn(d_model, d_ff)
    b1 = np.random.randn(d_ff)
    w2 = np.random.randn(d_ff, d_model)
    b2 = np.random.randn(d_model)

    return w1, b1, w2, b2


def feed_forward(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray
) -> tuple[np.ndarray, dict]:
    """
    Executa a Feed-Forward Network.

    Entrada:
    - x: (batch_size, seq_len, d_model)

    Saída:
    - output: (batch_size, seq_len, d_model)
    - debug_info: intermediários da FFN
    """
    hidden_linear = x @ w1 + b1
    hidden_relu = relu(hidden_linear)
    output = hidden_relu @ w2 + b2

    debug_info = {
        "hidden_linear": hidden_linear,
        "hidden_relu": hidden_relu,
        "output": output,
    }

    return output, debug_info