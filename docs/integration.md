# Integration

[README](../README.md) · [Method](method.md) · [Targets](targets.md)

DLoss needs three things from a training step: the decoder logits, which
positions belong to the report, and the encoded targets of each example. It
does not change the model, and nothing is needed at inference.

## Standard causal language model

For a Hugging Face-style model whose `labels` are `-100` at prompt, image and
padding positions:

```python
from diagnostic_loss import DiagnosticLossConfig, EncodedTargetStore, causal_diagnostic_loss

store = EncodedTargetStore.from_jsonl("targets/encoded_targets.jsonl", tokenizer)
config = DiagnosticLossConfig(lambda_pos=0.001, lambda_neg=0.00025, tau=8.0)

for batch in loader:
    targets = store.for_batch(batch.pop("sample_ids"))
    outputs = model(**batch)
    diagnostic = causal_diagnostic_loss(outputs.logits, batch["labels"], targets, config)
    loss = outputs.loss + diagnostic.loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

`causal_diagnostic_loss` shifts logits and labels by one position, as the
model's own cross-entropy does, and uses the positions whose label is not
`-100`. An optional `report_mask` aligned with `labels` restricts them further,
for example to exclude a fixed answer prefix.

`diagnostic.loss` is the sum of `diagnostic.positive_term` and
`diagnostic.negative_term`.

## Coefficients

The settings for each backbone in the paper are in
[`configs/paper/`](../configs/paper); all use `tau=8`.

| Backbone | Decoder | `lambda_pos` | `lambda_neg` |
| --- | --- | --- | --- |
| Merlin | RadLLaMA-7B | 0.001 | 0.0005 |
| fVLM | Qwen2.5-0.5B | 0.001 | 0.00025 |
| BTB3D | Llama-3.1-8B-Instruct | 0.001 | 0.00025 |
| HLIP | Qwen2.5-0.5B | 0.001 | 0.00025 |
| RadFM | MedLLaMA-13B | 0.01 | 0.01 |
| SPECTRE | Qwen2.5-0.5B | 0.01 | 0.005 |

These were chosen by a small grid search per backbone. For a new model,
`lambda_pos=0.001, lambda_neg=0.00025` (used for three of the six backbones) is a
reasonable first setting; compare cross-entropy against a run without DLoss to
check the diagnostic term is not dominating.

## Visual prefix

If image embeddings are concatenated in front of the text embeddings, extend
the labels with ignored positions so they line up with the logits:

```python
from diagnostic_loss import labels_with_visual_prefix

labels = labels_with_visual_prefix(text_labels, prefix_length=visual_embeds.shape[1])
outputs = model(inputs_embeds=torch.cat([visual_embeds, text_embeds], dim=1), labels=labels)
diagnostic = causal_diagnostic_loss(outputs.logits, labels, targets, config)
```

Use `after_bos=True` if the model inserts the prefix after the first text token.
[`examples/huggingface_qwen.py`](../examples/huggingface_qwen.py) runs this with
Qwen2.5-0.5B.

## Other layouts

When image tokens are interleaved with text, or the model returns logits in a
different layout, build the aligned inputs yourself and call `diagnostic_loss`
directly. `logits[i, t]` must predict the report token at position `t`, and
`report_mask[i, t]` must be true only at report positions:

```python
from diagnostic_loss import diagnostic_loss, report_cross_entropy

ce = report_cross_entropy(logits, labels, report_mask)
diagnostic = diagnostic_loss(logits, report_mask, targets, config)
loss = ce + diagnostic.loss
```

[`examples/custom_layout.py`](../examples/custom_layout.py) shows this with image
placeholder tokens and checks that non-report positions get no gradient. Print
`labels[report_mask]` for one batch before training to check the alignment.

For many concepts per example, lower `concept_chunk_size` (default 32) to reduce
peak memory.

## Distributed training and gradient accumulation

`diagnostic_loss` returns the mean over the examples with targets **in the
batch it is given**. Under DDP, gradients are averaged over ranks, so equal
weighting of each rank's mean is only correct when every rank has the same
number of examples with targets. To get the mean over the global batch, scale
each local loss by its share:

```python
import torch
import torch.distributed as dist

diagnostic = causal_diagnostic_loss(outputs.logits, labels, targets, config)
global_active = torch.tensor(diagnostic.active_samples, device=diagnostic.loss.device)
dist.all_reduce(global_active)
scale = dist.get_world_size() * diagnostic.active_samples / global_active.clamp_min(1)
loss = ce + diagnostic.loss * scale
```

Every rank must take part in the `all_reduce`, including ranks whose batch has
no targets. With gradient accumulation, count `global_active` over all
micro-batches of the optimizer step (it can be computed from the target rows
before the forward pass) and do not divide the diagnostic term by the number of
accumulation steps. Cross-entropy has its own denominator (report tokens) and
needs its own scale if you want an exact global mean.
