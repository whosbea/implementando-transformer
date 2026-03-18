import numpy as np
from math_utils import softmax


def initialize_attention_weights(d_model: int, d_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inicializa as matrizes de projeção W_Q, W_K e W_V.

    Shapes:
    - W_Q: (d_model, d_k)
    - W_K: (d_model, d_k)
    - W_V: (d_model, d_k)
    """
    w_q = np.random.randn(d_model, d_k)
    w_k = np.random.randn(d_model, d_k)
    w_v = np.random.randn(d_model, d_k)
    return w_q, w_k, w_v


def project_qkv(
    x_q: np.ndarray,
    x_k: np.ndarray,
    x_v: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Projeta entradas em Q, K e V.

    Permite:
    - self-attention: x_q = x_k = x_v
    - cross-attention: x_q diferente de x_k e x_v
    """
    q = x_q @ w_q
    k = x_k @ w_k
    v = x_v @ w_v
    return q, k, v


def scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None = None
) -> tuple[np.ndarray, dict]:
    """
    Calcula atenção escalada.

    Fórmula:
    Attention(Q, K, V) = softmax((QK^T / sqrt(d_k)) + M) V

    Entrada:
    - q: (batch_size, q_len, d_k)
    - k: (batch_size, k_len, d_k)
    - v: (batch_size, k_len, d_k)
    - mask: (q_len, k_len) ou None

    Saída:
    - output: (batch_size, q_len, d_k)
    - debug_info: tensores intermediários
    """
    d_k = q.shape[-1]

    # Transpor K nos dois últimos eixos
    k_transposed = np.transpose(k, (0, 2, 1))

    # Scores
    scores = q @ k_transposed

    # Escalonamento
    scaled_scores = scores / np.sqrt(d_k)

    # Aplicação opcional da máscara
    if mask is not None:
        scaled_scores = scaled_scores + mask

    # Softmax
    attention_weights = softmax(scaled_scores, axis=-1)

    # Saída
    output = attention_weights @ v

    debug_info = {
        "scores": scores,
        "scaled_scores": scaled_scores,
        "attention_weights": attention_weights,
        "output": output,
    }

    return output, debug_info