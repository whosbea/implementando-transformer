import numpy as np

from attention import initialize_attention_weights, project_qkv, scaled_dot_product_attention
from feed_forward import initialize_ffn_weights, feed_forward
from math_utils import layer_norm


def initialize_encoder_layer_weights(d_model: int, d_k: int, d_ff: int) -> dict:
    """
    Inicializa todos os pesos necessários para uma camada do encoder.
    """
    w_q, w_k, w_v = initialize_attention_weights(d_model, d_k)
    w1, b1, w2, b2 = initialize_ffn_weights(d_model, d_ff)

    return {
        "w_q": w_q,
        "w_k": w_k,
        "w_v": w_v,
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
    }


def encoder_layer(x: np.ndarray, layer_weights: dict) -> tuple[np.ndarray, dict]:
    """
    Executa uma camada completa do encoder.

    Fluxo:
    1. Self-Attention
    2. Add & Norm
    3. Feed-Forward
    4. Add & Norm
    """
    # Self-attention
    q, k, v = project_qkv(
        x, x, x,
        layer_weights["w_q"],
        layer_weights["w_k"],
        layer_weights["w_v"]
    )

    attention_output, attention_debug = scaled_dot_product_attention(q, k, v, mask=None)

    # Como a atenção sai com d_k, projetamos de volta para d_model
    # Neste laboratório vamos usar d_k = 64 e d_model = 128,
    # então precisamos compatibilizar a soma residual.
    if attention_output.shape[-1] != x.shape[-1]:
        raise ValueError(
            f"A saída da atenção tem shape {attention_output.shape}, "
            f"mas a entrada tem shape {x.shape}. "
            "Para residual, d_k deve ser igual a d_model "
            "ou deve existir uma projeção de saída."
        )

    # Primeiro Add & Norm
    x_res1 = x + attention_output
    x_norm1 = layer_norm(x_res1)

    # FFN
    ffn_output, ffn_debug = feed_forward(
        x_norm1,
        layer_weights["w1"],
        layer_weights["b1"],
        layer_weights["w2"],
        layer_weights["b2"],
    )

    # Segundo Add & Norm
    x_res2 = x_norm1 + ffn_output
    x_out = layer_norm(x_res2)

    debug_info = {
        "q": q,
        "k": k,
        "v": v,
        "attention_output": attention_output,
        "x_res1": x_res1,
        "x_norm1": x_norm1,
        "ffn_output": ffn_output,
        "x_res2": x_res2,
        "x_out": x_out,
        "attention_debug": attention_debug,
        "ffn_debug": ffn_debug,
    }

    return x_out, debug_info


def initialize_encoder_stack(n_layers: int, d_model: int, d_k: int, d_ff: int) -> list[dict]:
    """
    Inicializa os pesos de todas as camadas do encoder.
    """
    return [
        initialize_encoder_layer_weights(d_model, d_k, d_ff)
        for _ in range(n_layers)
    ]


def encoder_stack(x: np.ndarray, layers: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """
    Executa a pilha de camadas do encoder.
    """
    all_debug_info = []

    for layer_index, layer_weights in enumerate(layers, start=1):
        x, debug_info = encoder_layer(x, layer_weights)
        debug_info["layer_index"] = layer_index
        all_debug_info.append(debug_info)

    return x, all_debug_info