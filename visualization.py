import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def ensure_output_dir(output_dir: str = "outputs") -> None:
    """
    Garante que a pasta de saída exista.
    """
    os.makedirs(output_dir, exist_ok=True)


def draw_box(ax, x, y, w, h, text, fontsize=11):
    """
    Desenha uma caixa arredondada com texto centralizado.
    """
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        fill=False,
        linewidth=1.8
    )
    ax.add_patch(box)

    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True
    )


def draw_arrow(ax, x1, y1, x2, y2):
    """
    Desenha uma seta entre dois pontos.
    """
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", linewidth=1.8)
    )


def plot_transformer_overview(
    output_dir: str = "outputs",
    filename: str = "transformer_overview.png",
    show: bool = True
) -> None:
    """
    Plota a visão geral do Transformer completo.
    """
    ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Linha superior: fluxo da entrada fonte
    draw_box(ax, 0.6, 3.8, 2.4, 1.0, "src_embeddings\n(1, 6, 128)", fontsize=11)
    draw_box(ax, 3.8, 3.8, 2.8, 1.0, "Encoder Stack\n2 camadas", fontsize=11)
    draw_box(ax, 7.4, 3.8, 2.8, 1.0, "encoder_output\n(1, 6, 128)", fontsize=11)

    draw_arrow(ax, 3.0, 4.3, 3.8, 4.3)
    draw_arrow(ax, 6.6, 4.3, 7.4, 4.3)

    # Linha inferior: fluxo da entrada alvo
    draw_box(ax, 0.6, 1.4, 2.4, 1.0, "tgt_embeddings\n(1, 5, 128)", fontsize=11)
    draw_box(ax, 3.8, 1.4, 2.8, 1.0, "Decoder Stack\n2 camadas", fontsize=11)
    draw_box(ax, 7.4, 1.4, 2.8, 1.0, "decoder_output\n(1, 5, 128)", fontsize=11)
    draw_box(ax, 11.0, 1.4, 2.8, 1.0, "Projeção Final\n128 → 9", fontsize=11)
    draw_box(ax, 14.6, 1.4, 2.4, 1.0, "logits\n(1, 5, 9)", fontsize=11)

    draw_arrow(ax, 3.0, 1.9, 3.8, 1.9)
    draw_arrow(ax, 6.6, 1.9, 7.4, 1.9)
    draw_arrow(ax, 10.2, 1.9, 11.0, 1.9)
    draw_arrow(ax, 13.8, 1.9, 14.6, 1.9)

    # Ligação encoder -> decoder
    draw_arrow(ax, 8.8, 3.8, 8.8, 2.4)

    ax.text(
        9,
        5.4,
        "Arquitetura Integrada do Transformer Completo",
        ha="center",
        va="center",
        fontsize=16
    )

    ax.text(
        9,
        0.5,
        "O encoder processa a sequência fonte; o decoder usa a saída do encoder e projeta o resultado para o vocabulário.",
        ha="center",
        va="center",
        fontsize=10
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=220, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()


def plot_transformer_blocks(
    output_dir: str = "outputs",
    filename: str = "transformer_blocks.png",
    show: bool = True
) -> None:
    """
    Plota os blocos/camadas do Transformer completo.
    """
    ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Encoder
    draw_box(ax, 1.0, 3.5, 4.0, 1.2, "Encoder Layer 1\nSelf-Attention + Add&Norm + FFN + Add&Norm", fontsize=10)
    draw_box(ax, 1.0, 1.7, 4.0, 1.2, "Encoder Layer 2\nSelf-Attention + Add&Norm + FFN + Add&Norm", fontsize=10)

    draw_arrow(ax, 3.0, 3.5, 3.0, 2.9)

    # Decoder
    draw_box(ax, 7.0, 3.5, 4.2, 1.2, "Decoder Layer 1\nMasked Self-Attention + Cross-Attention + FFN", fontsize=10)
    draw_box(ax, 7.0, 1.7, 4.2, 1.2, "Decoder Layer 2\nMasked Self-Attention + Cross-Attention + FFN", fontsize=10)

    draw_arrow(ax, 9.1, 3.5, 9.1, 2.9)

    # Projeção final
    draw_box(ax, 13.2, 2.6, 3.2, 1.2, "Camada Linear Final\nDecoder Output → Vocab", fontsize=10)

    # Fluxos laterais
    draw_arrow(ax, 5.0, 4.1, 7.0, 4.1)
    draw_arrow(ax, 11.2, 3.2, 13.2, 3.2)

    # Ligação encoder -> decoder
    draw_arrow(ax, 5.0, 2.3, 7.0, 2.3)

    ax.text(
        9,
        5.4,
        "Blocos e Camadas Implementados no Transformer",
        ha="center",
        va="center",
        fontsize=16
    )

    ax.text(
        9,
        0.6,
        "Encoder e decoder possuem 2 camadas cada; a saída do decoder é projetada para o tamanho do vocabulário.",
        ha="center",
        va="center",
        fontsize=10
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=220, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()


def plot_transformer_inference_flow(
    output_dir: str = "outputs",
    filename: str = "transformer_inference_flow.png",
    show: bool = True
) -> None:
    """
    Plota o fluxo da inferência fim a fim do Transformer.
    """
    ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(18, 5.8))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 5.8)
    ax.axis("off")

    y = 2.4
    w = 2.6
    h = 1.1

    boxes = [
        (0.5, y, "Entrada Fonte\nsrc_token_ids"),
        (3.5, y, "Encoder\ngera encoder_output"),
        (6.5, y, "Começar alvo\ncom <START>"),
        (9.5, y, "Transformer\nForward"),
        (12.5, y, "Argmax sobre\núltima posição"),
        (15.2, y, "Adicionar token\nà sequência"),
    ]

    for x, y_box, text in boxes:
        draw_box(ax, x, y_box, w, h, text, fontsize=11)

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + w
        y1 = boxes[i][1] + h / 2
        x2 = boxes[i + 1][0]
        y2 = boxes[i + 1][1] + h / 2
        draw_arrow(ax, x1, y1, x2, y2)

    # Loop de volta
    draw_arrow(ax, 16.5, 2.4, 16.5, 1.1)
    draw_arrow(ax, 16.5, 1.1, 10.8, 1.1)
    draw_arrow(ax, 10.8, 1.1, 10.8, 2.4)

    # Texto de parada
    ax.text(
        16.5,
        4.4,
        "Parar se gerar\n<EOS>",
        ha="center",
        va="center",
        fontsize=10
    )

    ax.text(
        13.6,
        0.6,
        "Se não for <EOS>, repete",
        ha="center",
        va="center",
        fontsize=10
    )

    ax.text(
        9,
        5.2,
        "Fluxo da Inferência Fim a Fim no Transformer",
        ha="center",
        va="center",
        fontsize=16
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=220, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()