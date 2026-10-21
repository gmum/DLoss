import pytest
import torch

from diagnostic_loss import (
    ConceptTarget,
    DiagnosticLossConfig,
    SampleTargets,
    causal_diagnostic_loss,
    diagnostic_loss,
    labels_with_visual_prefix,
    report_cross_entropy,
)

CONFIG = DiagnosticLossConfig(lambda_pos=0.001, lambda_neg=0.00025, tau=8.0)


def test_causal_adapter_shifts_once_and_excludes_prompt():
    logits = torch.randn(1, 5, 7, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[-100, -100, 2, 3, -100]])
    targets = [SampleTargets("one", (ConceptTarget("a", (2,)),))]
    report_mask = torch.tensor([[0, 0, 0, 1, 0]], dtype=torch.bool)
    actual = causal_diagnostic_loss(logits, labels, targets, CONFIG, report_mask=report_mask)
    expected = diagnostic_loss(
        logits[:, :-1], torch.tensor([[0, 0, 1, 0]], dtype=torch.bool), targets, CONFIG
    )
    torch.testing.assert_close(actual.loss, expected.loss)
    gradient = torch.autograd.grad(actual.loss, logits)[0]
    assert torch.count_nonzero(gradient[:, [0, 1, 3, 4]]) == 0


def test_visual_prefix_labels_match_both_decoder_insertions():
    labels = torch.tensor([[-100, -100, 3, 4]])
    assert labels_with_visual_prefix(labels, 2).tolist() == [[-100, -100, -100, -100, 3, 4]]
    assert labels_with_visual_prefix(labels, 2, after_bos=True).tolist() == [
        [-100, -100, -100, -100, 3, 4]
    ]
    distinct = torch.tensor([[1, -100, 3, 4]])
    assert labels_with_visual_prefix(distinct, 2).tolist() == [[-100, -100, 1, -100, 3, 4]]
    assert labels_with_visual_prefix(distinct, 2, after_bos=True).tolist() == [
        [1, -100, -100, -100, 3, 4]
    ]
    with pytest.raises(ValueError, match="nonnegative"):
        labels_with_visual_prefix(labels, -1)


def test_report_cross_entropy_matches_paper_token_mean():
    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]], [[1.0, 0.0], [5.0, 5.0]]])
    labels = torch.tensor([[0, 1], [0, -100]])
    mask = labels != -100
    actual = report_cross_entropy(logits, labels, mask)
    expected = torch.nn.functional.cross_entropy(logits[mask], labels[mask])
    torch.testing.assert_close(actual, expected)


def test_report_cross_entropy_rejects_empty_mask():
    with pytest.raises(ValueError, match="at least one"):
        report_cross_entropy(
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, dtype=torch.long),
            torch.zeros(1, 1, dtype=torch.bool),
        )
