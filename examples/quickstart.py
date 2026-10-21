"""Train a tiny decoder with cross-entropy + Diagnostic Loss. Runs offline on CPU.

The reports, tokenizer and model are toys; the point is the sequence of API calls:
vocabulary -> target rows -> encoded targets -> loss in the training step.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from diagnostic_loss import (
    DiagnosticLossConfig,
    DiagnosticVocabulary,
    EncodedTargetStore,
    causal_diagnostic_loss,
    encode_target_specs,
)

# (sample_id, organ, reference finding, supported concepts)
FINDINGS = [
    ("case-1::liver", "liver", "fatty liver with a small cyst", ["steatosis", "cyst"]),
    ("case-1::kidneys", "kidneys", "normal kidneys", ["normal"]),
    ("case-2::gallbladder", "gallbladder", "gallstones in the gallbladder", ["gallstones"]),
    ("case-2::liver", "liver", "normal liver", ["normal"]),
    ("case-3::kidneys", "kidneys", "a stone in the left kidney", ["stone"]),
    ("case-3::gallbladder", "gallbladder", "sludge in the gallbladder", ["sludge"]),
]


class WordTokenizer:
    """Whitespace tokenizer exposing the two methods the encoder needs."""

    def __init__(self, texts: list[str]) -> None:
        words = sorted({word for text in texts for word in text.split()})
        self.vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, ":": 3}
        self.vocab.update({word: i + 4 for i, word in enumerate(words)})
        self.all_special_ids = [0, 1, 2]

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [self.vocab[word] for word in text.split()]

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)


class TinyDecoder(nn.Module):
    def __init__(self, vocabulary_size: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocabulary_size, 32)
        self.rnn = nn.GRU(32, 64, batch_first=True)
        self.head = nn.Linear(64, vocabulary_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.rnn(self.embed(input_ids))[0])


def make_batch(tokenizer: WordTokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    """Prompt '<bos> organ :' followed by the report; prompt labels are -100."""

    rows, labels = [], []
    for _, organ, report, _ in FINDINGS:
        prompt = [1, *tokenizer.encode(organ), 3]
        answer = [*tokenizer.encode(report), 2]
        rows.append(prompt + answer)
        labels.append([-100] * len(prompt) + answer)
    width = max(len(row) for row in rows)
    input_ids = torch.tensor([row + [0] * (width - len(row)) for row in rows])
    label_ids = torch.tensor([row + [-100] * (width - len(row)) for row in labels])
    return input_ids, label_ids


def main() -> None:
    torch.manual_seed(0)
    vocabulary = DiagnosticVocabulary.from_json(Path(__file__).with_name("example_vocabulary.json"))
    phrases = [p for concepts in vocabulary.organs.values() for ps in concepts.values() for p in ps]
    tokenizer = WordTokenizer(
        [text for _, organ, text, _ in FINDINGS] + list(vocabulary.organs) + phrases
    )

    # 1. One target row per organ finding: the supported concepts, and every
    #    other concept of that organ as an unsupported alternative.
    specs = [
        vocabulary.target_spec(sample_id, organ, supported)
        for sample_id, organ, _, supported in FINDINGS
    ]
    # 2. Encode once with the decoder's tokenizer, then look rows up by sample ID.
    rows, audit = encode_target_specs(specs, tokenizer)
    store = EncodedTargetStore(rows)
    print(
        f"encoded {audit.sample_count} rows: {audit.positive_concepts} supported, "
        f"{audit.negative_concepts} unsupported concepts"
    )

    input_ids, labels = make_batch(tokenizer)
    sample_ids = [sample_id for sample_id, *_ in FINDINGS]
    model = TinyDecoder(len(tokenizer.vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    # Coefficients are larger than in the paper so the effect shows in a few steps.
    config = DiagnosticLossConfig(lambda_pos=0.1, lambda_neg=0.05, tau=8.0)
    ce_loss = nn.CrossEntropyLoss()

    # 3. In the training step, add the diagnostic term to cross-entropy.
    for step in range(61):
        targets = store.for_batch(sample_ids)
        logits = model(input_ids)
        ce = ce_loss(logits[:, :-1].flatten(0, 1), labels[:, 1:].flatten())
        diagnostic = causal_diagnostic_loss(logits, labels, targets, config)
        loss = ce + diagnostic.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 20 == 0:
            print(
                f"step {step:2d}  ce {ce.item():.3f}  "
                f"supported {diagnostic.positive_term.item():+.4f}  "
                f"unsupported {diagnostic.negative_term.item():+.4f}"
            )


if __name__ == "__main__":
    main()
