"""Diagnostic Loss: asymmetric supervision for radiology report generation."""

from .alignment import causal_diagnostic_loss, labels_with_visual_prefix, report_cross_entropy
from .loss import (
    DiagnosticLossConfig,
    DiagnosticLossResult,
    diagnostic_loss,
    reference_diagnostic_loss,
)
from .targets import (
    ConceptTarget,
    EncodedTargetStore,
    EncodingAudit,
    LexicalConcept,
    SampleTargets,
    TargetSpec,
    encode_target_specs,
    load_encoded_targets,
    load_target_specs,
    save_encoded_targets,
    save_target_specs,
    tokenizer_fingerprint,
)
from .vocabulary import DiagnosticVocabulary

__version__ = "0.1.0"

__all__ = [
    "ConceptTarget",
    "DiagnosticLossConfig",
    "DiagnosticLossResult",
    "DiagnosticVocabulary",
    "EncodedTargetStore",
    "EncodingAudit",
    "LexicalConcept",
    "SampleTargets",
    "TargetSpec",
    "causal_diagnostic_loss",
    "diagnostic_loss",
    "encode_target_specs",
    "labels_with_visual_prefix",
    "load_encoded_targets",
    "load_target_specs",
    "reference_diagnostic_loss",
    "report_cross_entropy",
    "save_encoded_targets",
    "save_target_specs",
    "tokenizer_fingerprint",
]
