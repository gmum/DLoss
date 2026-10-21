"""Diagnostic Loss with an explicit report mask, for multimodal sequence layouts.

When image tokens are interleaved with text, build the mask of report positions
yourself and call ``diagnostic_loss`` directly.
"""

from __future__ import annotations

import torch
from torch import nn

from diagnostic_loss import (
    DiagnosticLossConfig,
    LexicalConcept,
    TargetSpec,
    diagnostic_loss,
    encode_target_specs,
    report_cross_entropy,
)


class PieceTokenizer:
    """Splits 'cysts' into two pieces, like a subword tokenizer would."""

    all_special_ids = (0, 1, 2, 3, 4)
    vocab = {
        "<pad>": 0,
        "<bos>": 1,
        "<image>": 2,
        "<ask>": 3,
        "<eos>": 4,
        "liver": 5,
        "kidney": 6,
        "cyst": 7,
        "##s": 8,
        "stone": 9,
    }
    pieces = {"cysts": ["cyst", "##s"], "stone": ["stone"]}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [self.vocab[piece] for word in text.split() for piece in self.pieces[word]]

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)


class TinyTransformer(nn.Module):
    def __init__(self, vocabulary_size: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocabulary_size, 16)
        layer = nn.TransformerEncoderLayer(16, 2, 32, dropout=0.0, batch_first=True)
        self.blocks = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.head = nn.Linear(16, vocabulary_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        causal = nn.Transformer.generate_square_subsequent_mask(input_ids.shape[1])
        return self.head(self.blocks(self.embed(input_ids), mask=causal, is_causal=True))


def main() -> None:
    torch.manual_seed(0)
    tokenizer = PieceTokenizer()
    cyst = LexicalConcept("cyst", ("cysts",))
    stone = LexicalConcept("stone", ("stone",))
    specs = [
        TargetSpec("case-1::liver", "liver finding", positive=(cyst,), negative=(stone,)),
        TargetSpec("case-1::kidney", "kidney finding", positive=(stone,), negative=(cyst,)),
    ]
    targets, _ = encode_target_specs(specs, tokenizer)
    assert targets[0].positive[0].token_ids == (7, 8)  # both pieces of "cysts"

    # <bos> <image> <image> <ask> organ | report ... <eos> <pad>
    input_ids = torch.tensor(
        [
            [1, 2, 2, 3, 5, 7, 8, 4, 0],
            [1, 2, 2, 3, 6, 9, 4, 0, 0],
        ]
    )
    is_report_token = torch.tensor(
        [
            [0, 0, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0],
        ],
        dtype=torch.bool,
    )

    model = TinyTransformer(len(tokenizer.vocab))
    logits = model(input_ids)

    # Align once: the logits at t predict the token at t + 1.
    aligned_logits = logits[:, :-1]
    aligned_labels = input_ids[:, 1:]
    report_mask = is_report_token[:, 1:]
    print("report tokens:", aligned_labels[report_mask].tolist())

    config = DiagnosticLossConfig(lambda_pos=0.001, lambda_neg=0.00025, tau=8.0)
    ce = report_cross_entropy(aligned_logits, aligned_labels, report_mask)
    diagnostic = diagnostic_loss(aligned_logits, report_mask, targets, config)
    aligned_logits.retain_grad()
    (ce + diagnostic.loss).backward()

    # Image, prompt and padding positions receive no gradient.
    assert aligned_logits.grad is not None
    assert torch.count_nonzero(aligned_logits.grad[~report_mask]) == 0
    print(
        f"ce {ce.item():.4f}  diagnostic {diagnostic.loss.item():+.6f}  "
        f"(active samples: {diagnostic.active_samples})"
    )


if __name__ == "__main__":
    main()
