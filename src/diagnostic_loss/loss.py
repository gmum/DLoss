"""Diagnostic Loss on aligned decoder logits.

``diagnostic_loss`` is the implementation to train with. ``reference_diagnostic_loss``
follows the paper's equations sample by sample and concept by concept; it is
slower and is kept for reading and for testing the batched version against.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .targets import ConceptTarget, SampleTargets


@dataclass(frozen=True)
class DiagnosticLossConfig:
    """Loss coefficients.

    ``lambda_pos`` and ``lambda_neg`` weight the supported- and unsupported-target
    terms, and ``tau`` is the log-sum-exp temperature for unsupported targets.
    ``epsilon`` clamps probabilities away from 0 and 1 before taking logarithms.
    """

    lambda_pos: float
    lambda_neg: float
    tau: float
    epsilon: float = 1e-6
    check_finite: bool = False

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.lambda_pos, self.lambda_neg, self.tau)):
            raise ValueError("lambda_pos, lambda_neg and tau must be finite")
        if self.lambda_pos < 0 or self.lambda_neg < 0:
            raise ValueError("lambda_pos and lambda_neg must be nonnegative")
        if self.tau <= 0:
            raise ValueError("tau must be positive")
        if not math.isfinite(self.epsilon) or not 0 < self.epsilon < 0.5:
            raise ValueError("epsilon must be in (0, 0.5)")


@dataclass(frozen=True)
class DiagnosticLossResult:
    """``loss == positive_term + negative_term``. Counts are for logging."""

    loss: torch.Tensor
    positive_term: torch.Tensor
    negative_term: torch.Tensor
    active_samples: int
    positive_concepts: int
    negative_concepts: int


def diagnostic_loss(
    logits: torch.Tensor,
    report_mask: torch.Tensor,
    targets: Sequence[SampleTargets],
    config: DiagnosticLossConfig,
    *,
    concept_chunk_size: int = 32,
) -> DiagnosticLossResult:
    """Compute Diagnostic Loss for a batch.

    Args:
        logits: ``[batch, positions, vocabulary]``. ``logits[i, t]`` must be the
            prediction for the report token at position ``t``.
        report_mask: bool ``[batch, positions]``, true at report positions. Prompt,
            image and padding positions must be false.
        targets: one row per batch element, in batch order. Rows without
            concepts contribute nothing and are excluded from the batch mean.
        config: loss coefficients.
        concept_chunk_size: concepts gathered at once; lower it to save memory.
    """

    _validate_inputs(logits, report_mask, targets, config)
    if concept_chunk_size < 1:
        raise ValueError("concept_chunk_size must be positive")

    values = _computation_dtype(logits)
    # Positions outside the mask may hold anything, including NaN placeholders.
    values = values.masked_fill(~report_mask[:, :, None], 0.0)
    zero = values.reshape(-1)[:1].nan_to_num().sum() * 0.0
    active = sum(sample.has_targets for sample in targets)
    positive_count = sum(len(sample.positive) for sample in targets)
    negative_count = sum(len(sample.negative) for sample in targets)
    if active == 0:
        return DiagnosticLossResult(zero, zero, zero, 0, 0, 0)

    ids, token_valid, weights, is_positive = _pack_concepts(targets, values.device, values.dtype)
    batch, positions, _ = values.shape
    max_concepts, max_tokens = ids.shape[1], ids.shape[2]
    sample_weights = torch.tensor(
        [sample.sample_weight for sample in targets], dtype=values.dtype, device=values.device
    )

    log_normalizer = torch.logsumexp(values, dim=-1)
    # Rows without report positions have no targets (checked above), but still
    # need one finite position so the masked log-sum-exp below is defined.
    lse_mask = report_mask.clone()
    lse_mask[:, 0] |= ~report_mask.any(dim=1)
    pos_sum = torch.zeros(batch, dtype=values.dtype, device=values.device)
    neg_sum = torch.zeros_like(pos_sum)
    pos_weight = torch.zeros_like(pos_sum)
    neg_weight = torch.zeros_like(pos_sum)

    for start in range(0, max_concepts, concept_chunk_size):
        stop = min(start + concept_chunk_size, max_concepts)
        chunk_ids = ids[:, start:stop]
        chunk_valid = token_valid[:, start:stop]
        selected = values.gather(
            2, chunk_ids.flatten(1)[:, None, :].expand(-1, positions, -1)
        ).reshape(batch, positions, stop - start, max_tokens)
        # m[i, t, d]: probability mass of concept d at position t.
        mass = ((selected - log_normalizer[:, :, None, None]).exp() * chunk_valid[:, None]).sum(-1)

        # Supported: noisy-OR over positions, L+ = -log(1 - prod_t (1 - m)).
        clamped = mass.clamp(max=1.0 - config.epsilon).masked_fill(~report_mask[:, :, None], 0.0)
        log_none = torch.log1p(-clamped).sum(dim=1)
        positive_loss = -torch.log((-torch.expm1(log_none)).clamp_min(config.epsilon))

        # Unsupported: L- = tau^-1 * log sum_t exp(tau * log m).
        log_mass = torch.log(mass.clamp_min(config.epsilon))
        negative_loss = (
            torch.logsumexp(
                (config.tau * log_mass).masked_fill(~lse_mask[:, :, None], -torch.inf), dim=1
            )
            / config.tau
        )

        w = weights[:, start:stop]
        w_pos = w * is_positive[:, start:stop]
        w_neg = w * ~is_positive[:, start:stop]
        pos_sum = pos_sum + (positive_loss * w_pos).sum(dim=1)
        neg_sum = neg_sum + (negative_loss * w_neg).sum(dim=1)
        pos_weight = pos_weight + w_pos.sum(dim=1)
        neg_weight = neg_weight + w_neg.sum(dim=1)

    # Weighted mean per branch; a branch with zero total weight contributes 0.
    row_pos = pos_sum / torch.where(pos_weight > 0, pos_weight, torch.ones_like(pos_weight))
    row_neg = neg_sum / torch.where(neg_weight > 0, neg_weight, torch.ones_like(neg_weight))
    positive_term = (sample_weights * config.lambda_pos * row_pos).sum() / active
    negative_term = (sample_weights * config.lambda_neg * row_neg).sum() / active
    return _finish(positive_term, negative_term, active, positive_count, negative_count, config)


def reference_diagnostic_loss(
    logits: torch.Tensor,
    report_mask: torch.Tensor,
    targets: Sequence[SampleTargets],
    config: DiagnosticLossConfig,
) -> DiagnosticLossResult:
    """Same contract and result as ``diagnostic_loss``, written as explicit loops."""

    _validate_inputs(logits, report_mask, targets, config)
    values = _computation_dtype(logits)
    zero = values[report_mask][:1].nan_to_num().sum() * 0.0
    positive_terms: list[torch.Tensor] = []
    negative_terms: list[torch.Tensor] = []

    for row, sample in enumerate(targets):
        if not sample.has_targets:
            continue
        row_logits = values[row, report_mask[row]]
        log_normalizer = torch.logsumexp(row_logits, dim=-1)
        positive = _branch(row_logits, log_normalizer, sample.positive, config, True, zero)
        negative = _branch(row_logits, log_normalizer, sample.negative, config, False, zero)
        positive_terms.append(sample.sample_weight * config.lambda_pos * positive)
        negative_terms.append(sample.sample_weight * config.lambda_neg * negative)

    active = len(positive_terms)
    positive_term = torch.stack(positive_terms).mean() if active else zero
    negative_term = torch.stack(negative_terms).mean() if active else zero
    return _finish(
        positive_term,
        negative_term,
        active,
        sum(len(sample.positive) for sample in targets),
        sum(len(sample.negative) for sample in targets),
        config,
    )


def _branch(
    row_logits: torch.Tensor,
    log_normalizer: torch.Tensor,
    concepts: tuple[ConceptTarget, ...],
    config: DiagnosticLossConfig,
    positive: bool,
    zero: torch.Tensor,
) -> torch.Tensor:
    weight_sum = sum(concept.weight for concept in concepts)
    if weight_sum == 0:
        return zero
    terms: list[torch.Tensor] = []
    for concept in concepts:
        if concept.weight == 0:
            continue
        log_mass = torch.logsumexp(row_logits[:, concept.token_ids], dim=-1) - log_normalizer
        if positive:
            mass = log_mass.exp().clamp(max=1.0 - config.epsilon)
            occurrence = -torch.expm1(torch.log1p(-mass).sum())
            concept_loss = -torch.log(occurrence.clamp_min(config.epsilon))
        else:
            clamped = log_mass.clamp_min(math.log(config.epsilon))
            concept_loss = torch.logsumexp(config.tau * clamped, dim=0) / config.tau
        terms.append(concept.weight * concept_loss)
    return torch.stack(terms).sum() / weight_sum if terms else zero


def _pack_concepts(
    targets: Sequence[SampleTargets], device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad every row's concepts into ``[batch, concepts, tokens]`` tensors."""

    batch = len(targets)
    max_concepts = max(len(sample.positive) + len(sample.negative) for sample in targets)
    max_tokens = max(
        len(concept.token_ids)
        for sample in targets
        for concept in (*sample.positive, *sample.negative)
    )
    ids = torch.zeros(batch, max_concepts, max_tokens, dtype=torch.long)
    token_valid = torch.zeros(batch, max_concepts, max_tokens, dtype=torch.bool)
    weights = torch.zeros(batch, max_concepts, dtype=dtype)
    is_positive = torch.zeros(batch, max_concepts, dtype=torch.bool)
    for row, sample in enumerate(targets):
        for col, concept in enumerate((*sample.positive, *sample.negative)):
            count = len(concept.token_ids)
            ids[row, col, :count] = torch.tensor(concept.token_ids, dtype=torch.long)
            token_valid[row, col, :count] = True
            weights[row, col] = concept.weight
            is_positive[row, col] = col < len(sample.positive)
    return ids.to(device), token_valid.to(device), weights.to(device), is_positive.to(device)


def _finish(
    positive_term: torch.Tensor,
    negative_term: torch.Tensor,
    active: int,
    positive_count: int,
    negative_count: int,
    config: DiagnosticLossConfig,
) -> DiagnosticLossResult:
    loss = positive_term + negative_term
    if config.check_finite and not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("diagnostic loss is non-finite")
    return DiagnosticLossResult(
        loss, positive_term, negative_term, active, positive_count, negative_count
    )


def _computation_dtype(logits: torch.Tensor) -> torch.Tensor:
    # Half-precision logits are evaluated in float32; float64 is kept for tests.
    return logits if logits.dtype == torch.float64 else logits.float()


def _validate_inputs(
    logits: torch.Tensor,
    report_mask: torch.Tensor,
    targets: Sequence[SampleTargets],
    config: DiagnosticLossConfig,
) -> None:
    if logits.ndim != 3 or min(logits.shape) == 0:
        raise ValueError("logits must have nonempty [batch, positions, vocabulary] shape")
    if not logits.is_floating_point():
        raise TypeError("logits must be floating point")
    if report_mask.shape != logits.shape[:2] or report_mask.dtype != torch.bool:
        raise ValueError("report_mask must be bool [batch, positions]")
    if report_mask.device != logits.device:
        raise ValueError("report_mask and logits must be on the same device")
    if len(targets) != logits.shape[0]:
        raise ValueError("one target row is required per batch element")
    vocabulary = logits.shape[-1]
    for sample, row_mask in zip(targets, report_mask, strict=True):
        for concept in (*sample.positive, *sample.negative):
            if max(concept.token_ids) >= vocabulary:
                raise ValueError(
                    f"concept {concept.name!r} has an ID outside the logits vocabulary"
                )
        if sample.has_targets and not bool(row_mask.any().item()):
            raise ValueError(f"sample {sample.sample_id!r} has targets but no report positions")
    if config.check_finite and not bool(torch.isfinite(logits[report_mask]).all().item()):
        raise FloatingPointError("a report-position logit is non-finite")
    computation = torch.float64 if logits.dtype == torch.float64 else torch.float32
    if config.epsilon < torch.finfo(computation).eps:
        raise ValueError(
            f"epsilon must be at least {torch.finfo(computation).eps:g} for {computation}; "
            "a smaller clamp cannot keep probabilities below one"
        )
