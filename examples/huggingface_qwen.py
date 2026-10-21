"""One training step with a pretrained Hugging Face decoder (Qwen2.5-0.5B).

Random features stand in for a vision encoder; a trainable projector maps them
to a visual prefix in front of the text, and the decoder stays frozen.

    pip install -e ".[examples]"
    python examples/huggingface_qwen.py

Downloads Qwen/Qwen2.5-0.5B (about 1 GB) on first use.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from diagnostic_loss import (
    DiagnosticLossConfig,
    DiagnosticVocabulary,
    EncodedTargetStore,
    causal_diagnostic_loss,
    encode_target_specs,
    labels_with_visual_prefix,
)

MODEL = "Qwen/Qwen2.5-0.5B"


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
    model.requires_grad_(False)

    vocabulary = DiagnosticVocabulary.from_json(Path(__file__).with_name("example_vocabulary.json"))
    spec = vocabulary.target_spec("case-1::liver", "liver", ["cyst"])
    rows, audit = encode_target_specs([spec], tokenizer)
    store = EncodedTargetStore(rows)
    print(
        f"supported: {[c.name for c in spec.positive]}, "
        f"unsupported: {[c.name for c in spec.negative]}"
    )

    prompt = tokenizer.encode("Findings for the liver: ", add_special_tokens=False)
    report = tokenizer.encode("A small cyst in the right lobe.", add_special_tokens=False)
    report.append(tokenizer.eos_token_id)
    input_ids = torch.tensor([prompt + report], device=device)
    text_labels = torch.tensor([[-100] * len(prompt) + report], device=device)

    projector = nn.Linear(64, model.config.hidden_size).to(device)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-4)
    visual_features = torch.randn(1, 4, 64, device=device)

    visual_prefix = projector(visual_features)
    inputs_embeds = torch.cat([visual_prefix, model.get_input_embeddings()(input_ids)], dim=1)
    labels = labels_with_visual_prefix(text_labels, prefix_length=visual_prefix.shape[1])
    outputs = model(inputs_embeds=inputs_embeds, labels=labels)

    config = DiagnosticLossConfig(lambda_pos=0.001, lambda_neg=0.00025, tau=8.0)
    diagnostic = causal_diagnostic_loss(
        outputs.logits, labels, store.for_batch(["case-1::liver"]), config
    )
    loss = outputs.loss + diagnostic.loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(
        f"ce {outputs.loss.item():.4f}  supported {diagnostic.positive_term.item():+.6f}  "
        f"unsupported {diagnostic.negative_term.item():+.6f}"
    )


if __name__ == "__main__":
    main()
