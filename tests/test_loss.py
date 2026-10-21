import math

import pytest
import torch

from diagnostic_loss import (
    ConceptTarget,
    DiagnosticLossConfig,
    SampleTargets,
    diagnostic_loss,
    reference_diagnostic_loss,
    report_cross_entropy,
)

CONFIG = DiagnosticLossConfig(lambda_pos=1.0, lambda_neg=1.0, tau=2.0)
PAPER_CONFIG = DiagnosticLossConfig(lambda_pos=0.001, lambda_neg=0.00025, tau=8.0)


def row(positive=(), negative=(), sample_weight=1.0, sample_id="sample"):
    return SampleTargets(sample_id, tuple(positive), tuple(negative), sample_weight)


def test_hand_calculated_paper_equations():
    probabilities = torch.tensor([[[0.2, 0.3, 0.5], [0.5, 0.1, 0.4]]], dtype=torch.float64)
    logits = probabilities.log().requires_grad_()
    targets = [row((ConceptTarget("supported", (0,)),), (ConceptTarget("unsupported", (1,)),))]
    result = reference_diagnostic_loss(logits, torch.ones(1, 2, dtype=torch.bool), targets, CONFIG)
    expected_pos = -math.log(1 - (1 - 0.2) * (1 - 0.5))
    expected_neg = math.log(0.3**2 + 0.1**2) / 2
    torch.testing.assert_close(
        result.positive_term, torch.tensor(expected_pos, dtype=torch.float64)
    )
    torch.testing.assert_close(
        result.negative_term, torch.tensor(expected_neg, dtype=torch.float64)
    )
    torch.testing.assert_close(
        result.loss, torch.tensor(expected_pos + expected_neg, dtype=torch.float64)
    )
    result.loss.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("loss_fn", [reference_diagnostic_loss, diagnostic_loss])
def test_full_paper_objective_with_token_and_concept_weights(loss_fn):
    probabilities = torch.tensor(
        [
            [[0.2, 0.3, 0.5], [0.5, 0.1, 0.4]],
            [[0.1, 0.2, 0.7], [0.4, 0.3, 0.3]],
        ],
        dtype=torch.float64,
    )
    logits = probabilities.log()
    mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.bool)
    labels = torch.tensor([[0, 1], [2, -100]])
    targets = [
        row(
            (
                ConceptTarget("combined", (0, 1), 2.0),
                ConceptTarget("single", (0,), 1.0),
            ),
            (ConceptTarget("absent", (2,), 3.0),),
            sample_id="one",
        ),
        row((ConceptTarget("second", (2,)),), sample_weight=0.5, sample_id="two"),
    ]
    config = DiagnosticLossConfig(lambda_pos=0.4, lambda_neg=0.2, tau=2.0)
    ce = report_cross_entropy(logits, labels, mask)
    result = loss_fn(logits, mask, targets, config)

    expected_ce = -(math.log(0.2) + math.log(0.1) + math.log(0.7)) / 3
    row_one_positive = (2 * -math.log(0.8) - math.log(0.6)) / 3
    row_one_negative = math.log(0.5**2 + 0.4**2) / 2
    row_two_positive = -math.log(0.7)
    expected_positive = 0.4 * (row_one_positive + 0.5 * row_two_positive) / 2
    expected_negative = 0.2 * row_one_negative / 2
    torch.testing.assert_close(ce, torch.tensor(expected_ce, dtype=torch.float64))
    torch.testing.assert_close(
        result.positive_term, torch.tensor(expected_positive, dtype=torch.float64)
    )
    torch.testing.assert_close(
        result.negative_term, torch.tensor(expected_negative, dtype=torch.float64)
    )
    torch.testing.assert_close(
        ce + result.loss,
        torch.tensor(expected_ce + expected_positive + expected_negative, dtype=torch.float64),
    )
    assert result.active_samples == 2


@pytest.mark.parametrize("loss_fn", [reference_diagnostic_loss, diagnostic_loss])
@pytest.mark.parametrize("check_finite", [False, True])
def test_excluded_nonfinite_logits_have_no_value_or_gradient_effect(loss_fn, check_finite):
    logits = torch.tensor(
        [[[float("nan"), 0.0, 0.0], [0.2, -0.1, 0.3], [0.0, float("inf"), 0.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    mask = torch.tensor([[False, True, False]])
    targets = [row((ConceptTarget("present", (0,)),), (ConceptTarget("absent", (1,)),))]
    config = DiagnosticLossConfig(1.0, 1.0, 2.0, check_finite=check_finite)
    result = loss_fn(logits, mask, targets, config)
    expected = loss_fn(logits[:, 1:2], torch.ones(1, 1, dtype=torch.bool), targets, config)
    torch.testing.assert_close(result.loss, expected.loss)
    gradient = torch.autograd.grad(result.loss, logits)[0]
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient[:, [0, 2]]) == 0


def test_negative_is_unnormalized_logsumexp_and_can_be_zero():
    logits = torch.zeros(1, 2, 2, dtype=torch.float64)
    target = [row(negative=(ConceptTarget("unsupported", (0,)),))]
    config = DiagnosticLossConfig(lambda_pos=0, lambda_neg=1, tau=1)
    result = reference_diagnostic_loss(logits, torch.ones(1, 2, dtype=torch.bool), target, config)
    torch.testing.assert_close(result.loss, torch.tensor(0.0, dtype=torch.float64))


def test_zero_weight_sample_remains_in_batch_denominator():
    logits = torch.zeros(2, 1, 2, dtype=torch.float64)
    concept = ConceptTarget("supported", (0,))
    targets = [row((concept,), sample_id="one"), row((concept,), sample_weight=0, sample_id="two")]
    config = DiagnosticLossConfig(lambda_pos=1, lambda_neg=0, tau=8)
    result = reference_diagnostic_loss(logits, torch.ones(2, 1, dtype=torch.bool), targets, config)
    torch.testing.assert_close(result.loss, torch.tensor(math.log(2) / 2, dtype=torch.float64))
    assert result.active_samples == 2


def test_empty_branches_and_samples():
    logits = torch.zeros(2, 1, 2, requires_grad=True)
    target = [row(), row(negative=(ConceptTarget("absent", (0,), 0.0),), sample_id="two")]
    result = reference_diagnostic_loss(logits, torch.ones(2, 1, dtype=torch.bool), target, CONFIG)
    assert result.active_samples == 1
    assert result.positive_concepts == 0
    assert result.negative_concepts == 1
    assert result.loss.item() == 0
    result.loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_masked_positions_do_not_change_loss_or_gradient():
    torch.manual_seed(2)
    logits = torch.randn(1, 4, 5, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[False, True, True, False]])
    targets = [row((ConceptTarget("supported", (1, 2)),), (ConceptTarget("absent", (3,)),))]
    first = reference_diagnostic_loss(logits, mask, targets, CONFIG).loss
    first_grad = torch.autograd.grad(first, logits)[0]
    modified = logits.detach().clone()
    modified[0, 0] = 1000
    modified[0, 3] = -1000
    modified.requires_grad_()
    second = reference_diagnostic_loss(modified, mask, targets, CONFIG).loss
    second_grad = torch.autograd.grad(second, modified)[0]
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first_grad, second_grad)
    assert torch.count_nonzero(first_grad[:, [0, 3]]) == 0


def _scalar_oracle(logits, mask, targets, config):
    # Independent direct transcription with torch.softmax and explicit products.
    probabilities = logits.softmax(dim=-1)
    samples = []
    for index, target in enumerate(targets):
        if not target.has_targets:
            continue
        positions = probabilities[index, mask[index]]
        branches = []
        for concepts, positive in ((target.positive, True), (target.negative, False)):
            weights = sum(concept.weight for concept in concepts)
            if weights == 0:
                branches.append(logits.reshape(-1)[0] * 0)
                continue
            terms = []
            for concept in concepts:
                mass = positions[:, list(concept.token_ids)].sum(-1)
                if positive:
                    value = -(1 - torch.prod(1 - mass)).log()
                else:
                    value = torch.log(torch.sum(mass.pow(config.tau))) / config.tau
                terms.append(value * concept.weight)
            branches.append(torch.stack(terms).sum() / weights)
        samples.append(
            target.sample_weight
            * (config.lambda_pos * branches[0] + config.lambda_neg * branches[1])
        )
    return torch.stack(samples).mean()


@pytest.mark.parametrize("loss_fn", [reference_diagnostic_loss, diagnostic_loss])
def test_random_value_and_gradient_parity_with_independent_oracle(loss_fn):
    torch.manual_seed(4)
    logits = torch.randn(3, 5, 9, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 0], [0, 1, 1, 0, 0]], dtype=torch.bool)
    targets = [
        row(
            (ConceptTarget("a", (1, 2), 2), ConceptTarget("b", (3,), 1)),
            (ConceptTarget("c", (4, 5), 0.7),),
            sample_id="a",
        ),
        row(negative=(ConceptTarget("d", (6,), 1),), sample_id="b"),
        row((ConceptTarget("e", (7,), 1),), sample_weight=0.3, sample_id="c"),
    ]
    actual = loss_fn(logits, mask, targets, CONFIG).loss
    expected = _scalar_oracle(logits, mask, targets, CONFIG)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(
        torch.autograd.grad(actual, logits, retain_graph=True)[0],
        torch.autograd.grad(expected, logits)[0],
        rtol=1e-10,
        atol=1e-10,
    )


def test_gradcheck():
    logits = torch.randn(1, 3, 5, dtype=torch.float64, requires_grad=True)
    targets = [row((ConceptTarget("a", (1,)),), (ConceptTarget("b", (2,)),))]
    mask = torch.ones(1, 3, dtype=torch.bool)
    assert torch.autograd.gradcheck(
        lambda x: reference_diagnostic_loss(x, mask, targets, CONFIG).loss, (logits,)
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mixed_precision_uses_stable_computation(dtype):
    logits = torch.tensor([[[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]]], dtype=dtype, requires_grad=True)
    targets = [row((ConceptTarget("supported", (0,)),), (ConceptTarget("unsupported", (2,)),))]
    result = reference_diagnostic_loss(logits, torch.ones(1, 2, dtype=torch.bool), targets, CONFIG)
    assert result.loss.dtype == torch.float32
    result.loss.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_value_and_gradient_parity():
    torch.manual_seed(21)
    cpu = torch.randn(2, 4, 8, dtype=torch.float32, requires_grad=True)
    gpu = cpu.detach().cuda().requires_grad_()
    mask = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]], dtype=torch.bool)
    targets = [
        row((ConceptTarget("a", (1, 2)),), (ConceptTarget("b", (3,)),), sample_id="a"),
        row((ConceptTarget("c", (4,)),), sample_id="b"),
    ]
    cpu_loss = reference_diagnostic_loss(cpu, mask, targets, CONFIG).loss
    gpu_loss = reference_diagnostic_loss(gpu, mask.cuda(), targets, CONFIG).loss
    torch.testing.assert_close(cpu_loss, gpu_loss.cpu(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        torch.autograd.grad(cpu_loss, cpu)[0],
        torch.autograd.grad(gpu_loss, gpu)[0].cpu(),
        rtol=1e-4,
        atol=1e-5,
    )


def test_invalid_inputs():
    with pytest.raises(ValueError, match="no report positions"):
        reference_diagnostic_loss(
            torch.zeros(1, 2, 3),
            torch.zeros(1, 2, dtype=torch.bool),
            [row((ConceptTarget("a", (1,)),))],
            CONFIG,
        )
    with pytest.raises(ValueError, match="outside the logits vocabulary"):
        reference_diagnostic_loss(
            torch.zeros(1, 2, 3),
            torch.ones(1, 2, dtype=torch.bool),
            [row((ConceptTarget("a", (3,)),))],
            CONFIG,
        )
    with pytest.raises(ValueError, match="outside the logits vocabulary"):
        reference_diagnostic_loss(
            torch.zeros(1, 2, 3),
            torch.ones(1, 2, dtype=torch.bool),
            [row((ConceptTarget("zero-weight", (3,), 0),))],
            CONFIG,
        )
    with pytest.raises(ValueError, match="duplicate"):
        ConceptTarget("a", (1, 1))
    with pytest.raises(ValueError, match="positive"):
        DiagnosticLossConfig(lambda_pos=1, lambda_neg=1, tau=0)


@pytest.mark.parametrize("active_counts", [(1, 2), (0, 2), (0, 0)])
@pytest.mark.parametrize("world_size", [1, 2])
def test_partition_scaling_preserves_global_value_and_gradient(active_counts, world_size):
    # Each rank has two microbatches. Emulate DDP's gradient mean explicitly;
    # world_size=1 also checks ordinary gradient accumulation.
    torch.manual_seed(19)
    logits = torch.randn(world_size * 4, 2, 3, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(logits.shape[:2], dtype=torch.bool)
    targets = []
    for rank in range(world_size):
        for count in active_counts:
            local_count = max(count - rank, 0)
            for index in range(2):
                targets.append(
                    row(
                        (ConceptTarget("p", (0,), 0.5),) if index < local_count else (),
                        (ConceptTarget("n", (1,), 2.0),) if index < local_count else (),
                        sample_weight=0.0 if rank == 1 and index == 0 else 0.7,
                        sample_id=str(len(targets)),
                    )
                )
    expected = diagnostic_loss(logits, mask, targets, CONFIG).loss
    global_active = sum(target.has_targets for target in targets)
    scaled_terms = []
    for start in range(0, len(targets), 2):
        result = diagnostic_loss(
            logits[start : start + 2], mask[start : start + 2], targets[start : start + 2], CONFIG
        )
        scale = world_size * result.active_samples / max(global_active, 1)
        scaled_terms.append(result.loss * scale)
    actual = torch.stack(scaled_terms).sum() / world_size
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(
        torch.autograd.grad(actual, logits, retain_graph=True)[0],
        torch.autograd.grad(expected, logits)[0],
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize("loss_fn", [reference_diagnostic_loss, diagnostic_loss])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_epsilon_matches_computation_precision(loss_fn, dtype):
    logits = torch.tensor([[[100.0, -100.0]]], dtype=dtype, requires_grad=True)
    targets = [row((ConceptTarget("saturated", (0,)),))]
    mask = torch.ones(1, 1, dtype=torch.bool)
    config = DiagnosticLossConfig(1.0, 1.0, 8.0, epsilon=1e-12)
    if dtype != torch.float64:
        with pytest.raises(ValueError, match="epsilon must be at least"):
            loss_fn(logits, mask, targets, config)
    else:
        result = loss_fn(logits, mask, targets, config)
        result.loss.backward()
        assert torch.isfinite(logits.grad).all()
    logits.grad = None
    result = loss_fn(logits, mask, targets, CONFIG)
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("chunk_size", [1, 2, 16])
def test_batched_matches_reference_value_and_gradient(chunk_size):
    torch.manual_seed(31)
    reference_logits = torch.randn(4, 6, 13, dtype=torch.float64, requires_grad=True)
    batched_logits = reference_logits.detach().clone().requires_grad_()
    mask = torch.tensor(
        [
            [0, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0],
        ],
        dtype=torch.bool,
    )
    targets = [
        SampleTargets(
            "a",
            (
                ConceptTarget("p1", (1, 2), 0.5),
                ConceptTarget("p2", (3,), 2.0),
            ),
            (ConceptTarget("n1", (4, 5), 1.0),),
        ),
        SampleTargets("b", negative=(ConceptTarget("n2", (6,), 0.0),), sample_weight=0),
        SampleTargets("empty"),
        SampleTargets(
            "d", (ConceptTarget("p3", (7, 8), 1.0),), (ConceptTarget("n3", (8, 9), 0.7),)
        ),
    ]
    reference = reference_diagnostic_loss(reference_logits, mask, targets, PAPER_CONFIG)
    batched = diagnostic_loss(
        batched_logits, mask, targets, PAPER_CONFIG, concept_chunk_size=chunk_size
    )
    for name in ("loss", "positive_term", "negative_term"):
        torch.testing.assert_close(
            getattr(batched, name), getattr(reference, name), rtol=1e-11, atol=1e-11
        )
    assert batched.active_samples == reference.active_samples
    assert batched.positive_concepts == reference.positive_concepts
    assert batched.negative_concepts == reference.negative_concepts
    torch.testing.assert_close(
        torch.autograd.grad(batched.loss, batched_logits)[0],
        torch.autograd.grad(reference.loss, reference_logits)[0],
        rtol=1e-10,
        atol=1e-10,
    )


def test_batched_rejects_empty_positions_for_active_row():
    with pytest.raises(ValueError, match="no report positions"):
        diagnostic_loss(
            torch.zeros(1, 2, 4),
            torch.zeros(1, 2, dtype=torch.bool),
            [SampleTargets("a", (ConceptTarget("p", (1,)),))],
            PAPER_CONFIG,
        )


def test_batched_mask_and_finite_check():
    logits = torch.zeros(1, 3, 4, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[False, True, False]])
    target = [SampleTargets("a", (ConceptTarget("p", (1,)),))]
    config = DiagnosticLossConfig(lambda_pos=1, lambda_neg=0, tau=8, check_finite=True)
    baseline = diagnostic_loss(logits, mask, target, config).loss
    gradient = torch.autograd.grad(baseline, logits)[0]
    assert torch.count_nonzero(gradient[:, [0, 2]]) == 0
    masked_nan = logits.detach().clone()
    masked_nan[0, 0, 0] = float("nan")
    masked_nan.requires_grad_()
    masked_result = diagnostic_loss(masked_nan, mask, target, config).loss
    torch.testing.assert_close(masked_result, baseline)
    masked_gradient = torch.autograd.grad(masked_result, masked_nan)[0]
    assert torch.isfinite(masked_gradient).all()
    assert torch.count_nonzero(masked_gradient[:, [0, 2]]) == 0
    valid_nan = logits.detach().clone()
    valid_nan[0, 1, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="report-position logit"):
        diagnostic_loss(valid_nan, mask, target, config)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_batched_mixed_precision_parity(dtype):
    logits = torch.randn(2, 4, 8, dtype=dtype)
    mask = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]], dtype=torch.bool)
    targets = [
        SampleTargets("a", (ConceptTarget("p", (1, 2)),), (ConceptTarget("n", (3,)),)),
        SampleTargets("b", negative=(ConceptTarget("n2", (4,)),)),
    ]
    actual = diagnostic_loss(logits, mask, targets, PAPER_CONFIG).loss
    expected = reference_diagnostic_loss(logits, mask, targets, PAPER_CONFIG).loss
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_batched_cuda_parity():
    logits = torch.randn(2, 5, 11, device="cuda", requires_grad=True)
    mask = torch.tensor([[1, 0, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool, device="cuda")
    targets = [
        SampleTargets("a", (ConceptTarget("p", (1, 2)),), (ConceptTarget("n", (3,)),)),
        SampleTargets("b", negative=(ConceptTarget("n2", (4,)),)),
    ]
    actual = diagnostic_loss(logits, mask, targets, PAPER_CONFIG).loss
    expected = reference_diagnostic_loss(logits, mask, targets, PAPER_CONFIG).loss
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        torch.autograd.grad(actual, logits, retain_graph=True)[0],
        torch.autograd.grad(expected, logits)[0],
        rtol=1e-4,
        atol=1e-5,
    )
