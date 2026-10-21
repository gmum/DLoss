"""Helpers for aligning causal language-model outputs with report positions."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .loss import DiagnosticLossConfig, DiagnosticLossResult, diagnostic_loss
from .targets import SampleTargets


def causal_diagnostic_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    targets: Sequence[SampleTargets],
    config: DiagnosticLossConfig,
    *,
    report_mask: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> DiagnosticLossResult:
    """Diagnostic Loss for standard (unshifted) causal-LM ``logits`` and ``labels``.

    Shifts by one position, as Hugging Face causal LMs do for cross-entropy, and
    uses positions whose label is not ``ignore_index``. An optional bool
    ``report_mask``, aligned with ``labels``, restricts the positions further.
    """

    if logits.ndim != 3 or labels.shape != logits.shape[:2] or logits.shape[1] < 2:
        raise ValueError("expected logits [B, T, V] and labels [B, T] with T >= 2")
    if report_mask is not None and (
        report_mask.shape != labels.shape or report_mask.dtype != torch.bool
    ):
        raise ValueError("report_mask must be bool [B, T], aligned with labels")
    mask = labels[:, 1:] != ignore_index
    if report_mask is not None:
        mask = mask & report_mask[:, 1:]
    return diagnostic_loss(logits[:, :-1, :], mask, targets, config)


def labels_with_visual_prefix(
    labels: torch.Tensor,
    prefix_length: int,
    *,
    after_bos: bool = False,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Insert ignored label positions for visual embeddings placed before the text.

    Use ``after_bos=True`` if the model inserts the visual prefix after the
    first text token rather than before it.
    """

    if labels.ndim != 2 or labels.shape[1] == 0:
        raise ValueError("labels must be nonempty [batch, text positions]")
    if prefix_length < 0:
        raise ValueError("prefix_length must be nonnegative")
    split = 1 if after_bos else 0
    ignored = torch.full(
        (labels.shape[0], prefix_length), ignore_index, dtype=labels.dtype, device=labels.device
    )
    return torch.cat([labels[:, :split], ignored, labels[:, split:]], dim=1)


def report_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    report_mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy averaged over report tokens, for already aligned inputs.

    ``logits[i, t]`` predicts ``labels[i, t]``; only positions where
    ``report_mask`` is true are used.
    """

    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("expected logits [B, T, V] and labels [B, T]")
    if not logits.is_floating_point():
        raise TypeError("logits must be floating point")
    if report_mask.shape != labels.shape or report_mask.dtype != torch.bool:
        raise ValueError("report_mask must be bool [B, T]")
    if labels.device != logits.device or report_mask.device != logits.device:
        raise ValueError("logits, labels and report_mask must be on the same device")
    if not bool(report_mask.any().item()):
        raise ValueError("cross-entropy needs at least one report token")
    values = logits if logits.dtype == torch.float64 else logits.float()
    selected = values[report_mask]
    selected_labels = labels[report_mask]
    if bool(((selected_labels < 0) | (selected_labels >= values.shape[-1])).any().item()):
        raise ValueError("a report label is outside the logits vocabulary")
    label_logits = selected.gather(1, selected_labels[:, None]).squeeze(1)
    return (torch.logsumexp(selected, dim=-1) - label_logits).mean()
