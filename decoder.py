import numpy as np

from attention import initialize_attention_weights, project_qkv, scaled_dot_product_attention
from feed_forward import initialize_ffn_weights, feed_forward
from math_utils import layer_norm
from masking import create_causal_mask


def initialize_decoder_layer_weights(d_model: int, d_k: int, d_ff: int) -> dict:
    """
    Inicializa todos os pesos de uma camada do decoder.
    """
    # Self-attention mascarada
    w_q_self, w_k_self, w_v_self = initialize_attention_weights(d_model, d_k)

    # Cross-attention
    w_q_cross, w_k_cross, w_v_cross = initialize_attention_weights(d_model, d_k)

    # FFN
    w1, b1, w2, b2 = initialize_ffn_weights(d_model, d_ff)

    return {
        "w_q_self": w_q_self,
        "w_k_self": w_k_self,
        "w_v_self": w_v_self,
        "w_q_cross": w_q_cross,
        "w_k_cross": w_k_cross,
        "w_v_cross": w_v_cross,
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
    }


def decoder_layer(
    decoder_input: np.ndarray,
    encoder_output: np.ndarray,
    layer_weights: dict
) -> tuple[np.ndarray, dict]:
    """
    Executa uma camada completa do decoder.

    Fluxo:
    1. Masked Self-Attention
    2. Add & Norm
    3. Cross-Attention
    4. Add & Norm
    5. FFN
    6. Add & Norm
    """
    batch_size, tgt_seq_len, d_model = decoder_input.shape

    # 1. Masked Self-Attention
    q_self, k_self, v_self = project_qkv(
        decoder_input,
        decoder_input,
        decoder_input,
        layer_weights["w_q_self"],
        layer_weights["w_k_self"],
        layer_weights["w_v_self"],
    )

    causal_mask = create_causal_mask(tgt_seq_len)

    masked_self_attention_output, masked_self_debug = scaled_dot_product_attention(
        q_self, k_self, v_self, mask=causal_mask
    )

    x_res1 = decoder_input + masked_self_attention_output
    x_norm1 = layer_norm(x_res1)

    # 2. Cross-Attention
    q_cross, k_cross, v_cross = project_qkv(
        x_norm1,
        encoder_output,
        encoder_output,
        layer_weights["w_q_cross"],
        layer_weights["w_k_cross"],
        layer_weights["w_v_cross"],
    )

    cross_attention_output, cross_attention_debug = scaled_dot_product_attention(
        q_cross, k_cross, v_cross, mask=None
    )

    x_res2 = x_norm1 + cross_attention_output
    x_norm2 = layer_norm(x_res2)

    # 3. Feed-Forward
    ffn_output, ffn_debug = feed_forward(
        x_norm2,
        layer_weights["w1"],
        layer_weights["b1"],
        layer_weights["w2"],
        layer_weights["b2"],
    )

    x_res3 = x_norm2 + ffn_output
    x_out = layer_norm(x_res3)

    debug_info = {
        "causal_mask": causal_mask,
        "masked_self_attention_output": masked_self_attention_output,
        "x_norm1": x_norm1,
        "cross_attention_output": cross_attention_output,
        "x_norm2": x_norm2,
        "ffn_output": ffn_output,
        "x_out": x_out,
        "masked_self_debug": masked_self_debug,
        "cross_attention_debug": cross_attention_debug,
        "ffn_debug": ffn_debug,
    }

    return x_out, debug_info


def initialize_decoder_stack(n_layers: int, d_model: int, d_k: int, d_ff: int) -> list[dict]:
    """
    Inicializa os pesos de todas as camadas do decoder.
    """
    return [
        initialize_decoder_layer_weights(d_model, d_k, d_ff)
        for _ in range(n_layers)
    ]


def decoder_stack(
    decoder_input: np.ndarray,
    encoder_output: np.ndarray,
    layers: list[dict]
) -> tuple[np.ndarray, list[dict]]:
    """
    Executa a pilha de camadas do decoder.
    """
    all_debug_info = []

    x = decoder_input
    for layer_index, layer_weights in enumerate(layers, start=1):
        x, debug_info = decoder_layer(x, encoder_output, layer_weights)
        debug_info["layer_index"] = layer_index
        all_debug_info.append(debug_info)

    return x, all_debug_info