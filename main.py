import numpy as np

from config import (
    SEED,
    D_MODEL,
    D_K,
    D_FF,
    N_ENCODER_LAYERS,
    N_DECODER_LAYERS,
    BATCH_SIZE,
    SRC_SEQ_LEN,
    TGT_SEQ_LEN,
    VOCAB_SIZE,
    VOCAB,
    ID_TO_TOKEN,
    START_TOKEN_ID,
    EOS_TOKEN_ID,
)
from attention import (
    initialize_attention_weights,
    project_qkv,
    scaled_dot_product_attention,
)
from feed_forward import initialize_ffn_weights, feed_forward
from encoder import initialize_encoder_stack, encoder_stack
from masking import create_causal_mask
from decoder import initialize_decoder_stack, decoder_stack
from transformer import initialize_transformer_weights, transformer_forward
from inference import create_token_embedding_table, tokens_to_embeddings, autoregressive_generate
from visualization import (
    plot_transformer_overview,
    plot_transformer_blocks,
    plot_transformer_inference_flow,
)

def main():
    print("=== LABORATÓRIO 4: TRANSFORMER COMPLETO ===")
    print(f"Seed: {SEED}")
    print(f"D_MODEL: {D_MODEL}")
    print(f"D_K: {D_K}")
    print(f"D_FF: {D_FF}")
    print(f"N_ENCODER_LAYERS: {N_ENCODER_LAYERS}")
    print(f"N_DECODER_LAYERS: {N_DECODER_LAYERS}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"SRC_SEQ_LEN: {SRC_SEQ_LEN}")
    print(f"TGT_SEQ_LEN: {TGT_SEQ_LEN}")
    print(f"VOCAB_SIZE: {VOCAB_SIZE}")

    x = np.random.randn(BATCH_SIZE, SRC_SEQ_LEN, D_MODEL)

    print("\n=== TESTE DA ATENÇÃO GENÉRICA ===")

    w_q, w_k, w_v = initialize_attention_weights(D_MODEL, D_K)
    q, k, v = project_qkv(x, x, x, w_q, w_k, w_v)

    attention_output, attention_debug = scaled_dot_product_attention(q, k, v, mask=None)

    print("Shape de x:", x.shape)
    print("Shape de Q:", q.shape)
    print("Shape de K:", k.shape)
    print("Shape de V:", v.shape)
    print("Shape dos scores:", attention_debug["scores"].shape)
    print("Shape dos scaled_scores:", attention_debug["scaled_scores"].shape)
    print("Shape dos attention_weights:", attention_debug["attention_weights"].shape)
    print("Shape da saída da atenção:", attention_output.shape)

    print("\nSoma das linhas dos attention_weights:")
    print(np.sum(attention_debug["attention_weights"], axis=-1))

    print("\n=== TESTE DA FEED-FORWARD NETWORK ===")

    w1, b1, w2, b2 = initialize_ffn_weights(D_MODEL, D_FF)
    ffn_output, ffn_debug = feed_forward(x, w1, b1, w2, b2)

    print("Shape da entrada da FFN:", x.shape)
    print("Shape após primeira linear:", ffn_debug["hidden_linear"].shape)
    print("Shape após ReLU:", ffn_debug["hidden_relu"].shape)
    print("Shape da saída da FFN:", ffn_output.shape)

    print("\n=== TESTE DO ENCODER COMPLETO ===")

    encoder_layers = initialize_encoder_stack(
        n_layers=N_ENCODER_LAYERS,
        d_model=D_MODEL,
        d_k=D_K,
        d_ff=D_FF,
    )

    encoder_output, encoder_debug = encoder_stack(x, encoder_layers)

    print("Shape da entrada do encoder:", x.shape)
    print("Número de camadas do encoder:", N_ENCODER_LAYERS)
    print("Shape da saída final do encoder:", encoder_output.shape)

    print("\n=== TESTE DA MÁSCARA CAUSAL ===")

    causal_mask = create_causal_mask(TGT_SEQ_LEN)
    print("Shape da máscara causal:", causal_mask.shape)
    print(causal_mask)

    print("\n=== TESTE DO DECODER COMPLETO ===")

    decoder_input = np.random.randn(BATCH_SIZE, TGT_SEQ_LEN, D_MODEL)

    decoder_layers = initialize_decoder_stack(
        n_layers=N_DECODER_LAYERS,
        d_model=D_MODEL,
        d_k=D_K,
        d_ff=D_FF,
    )

    decoder_output, decoder_debug = decoder_stack(
        decoder_input=decoder_input,
        encoder_output=encoder_output,
        layers=decoder_layers,
    )

    print("Shape da entrada do decoder:", decoder_input.shape)
    print("Número de camadas do decoder:", N_DECODER_LAYERS)
    print("Shape da saída final do decoder:", decoder_output.shape)

    print("\n=== TESTE DO TRANSFORMER COMPLETO ===")

    weights = initialize_transformer_weights(
        n_encoder_layers=N_ENCODER_LAYERS,
        n_decoder_layers=N_DECODER_LAYERS,
        d_model=D_MODEL,
        d_k=D_K,
        d_ff=D_FF,
        vocab_size=VOCAB_SIZE,
    )

    embedding_table = create_token_embedding_table(VOCAB_SIZE, D_MODEL)

    src_token_ids = [VOCAB["eu"], VOCAB["gosto"], VOCAB["de"], VOCAB["pinguins"], VOCAB["muito"], VOCAB["fim"]]
    tgt_token_ids = [START_TOKEN_ID, VOCAB["eu"], VOCAB["gosto"], VOCAB["de"], VOCAB["pinguins"]]

    src_embeddings = tokens_to_embeddings(src_token_ids, embedding_table)
    tgt_embeddings = tokens_to_embeddings(tgt_token_ids, embedding_table)

    logits, transformer_debug = transformer_forward(
        src_embeddings=src_embeddings,
        tgt_embeddings=tgt_embeddings,
        weights=weights,
    )

    print("Shape do src_embeddings:", src_embeddings.shape)
    print("Shape do tgt_embeddings:", tgt_embeddings.shape)
    print("Shape do encoder_output:", transformer_debug["encoder_output"].shape)
    print("Shape do decoder_output:", transformer_debug["decoder_output"].shape)
    print("Shape dos logits finais:", logits.shape)

    print("\n=== TESTE DA INFERÊNCIA FIM A FIM ===")

    generated_ids, all_probs, _ = autoregressive_generate(
        src_token_ids=src_token_ids,
        embedding_table=embedding_table,
        weights=weights,
        start_token_id=START_TOKEN_ID,
        eos_token_id=EOS_TOKEN_ID,
        max_len=8,
    )

    print("Sequência gerada (IDs):")
    print(generated_ids)

    generated_tokens = [ID_TO_TOKEN[token_id] for token_id in generated_ids]
    print("\nSequência gerada (tokens):")
    print(generated_tokens)

    print("\nDistribuições de probabilidade por passo:")
    for step, probs in enumerate(all_probs):
        print(f"\nPasso {step}:")
        print(probs)
        print("Soma das probabilidades:", np.sum(probs))

    print("\n=== VISUALIZAÇÕES DO TRANSFORMER ===")

    plot_transformer_overview(
        output_dir="outputs",
        filename="transformer_overview.png",
        show=True
    )

    plot_transformer_blocks(
        output_dir="outputs",
        filename="transformer_blocks.png",
        show=True
    )

    plot_transformer_inference_flow(
        output_dir="outputs",
        filename="transformer_inference_flow.png",
        show=True
    )

    print("Diagramas salvos em outputs/")

if __name__ == "__main__":
    main()