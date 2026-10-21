"""Diagnostic targets: lexical specifications, tokenizer encoding and storage.

A target row describes one text unit (in the paper, one organ finding). It lists
supported concepts and unsupported alternatives. Lexical rows (``TargetSpec``)
hold phrases; encoded rows (``SampleTargets``) hold the decoder vocabulary IDs
that the loss consumes.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

SPEC_VERSION = "diagnostic-loss-spec-v1"
ENCODED_VERSION = "diagnostic-loss-encoded-v1"


class Tokenizer(Protocol):
    """The subset of the Hugging Face tokenizer interface used for encoding."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> Sequence[int]: ...

    def get_vocab(self) -> Mapping[str, int]: ...


@dataclass(frozen=True)
class ConceptTarget:
    """One diagnostic concept as a set of unique decoder vocabulary IDs."""

    name: str
    token_ids: tuple[int, ...]
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("concept name must be nonempty")
        if not self.token_ids:
            raise ValueError(f"concept {self.name!r} needs at least one token ID")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.token_ids
        ):
            raise ValueError(f"concept {self.name!r} has an invalid token ID")
        if len(set(self.token_ids)) != len(self.token_ids):
            raise ValueError(f"concept {self.name!r} has duplicate token IDs")
        _check_weight(self.weight, f"concept {self.name!r} weight")


@dataclass(frozen=True)
class SampleTargets:
    """Encoded supported and unsupported concepts for one text unit."""

    sample_id: str
    positive: tuple[ConceptTarget, ...] = ()
    negative: tuple[ConceptTarget, ...] = ()
    sample_weight: float = 1.0
    text_unit: str = ""
    provenance: str = ""

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id must be nonempty")
        _check_weight(self.sample_weight, "sample_weight")
        if not isinstance(self.provenance, str):
            raise TypeError("provenance must be a string")
        _check_unique_names(concept.name for concept in (*self.positive, *self.negative))

    @property
    def has_targets(self) -> bool:
        return bool(self.positive or self.negative)


@dataclass(frozen=True)
class LexicalConcept:
    """A concept given by its lexical forms. Whitespace in phrases is preserved."""

    name: str
    phrases: tuple[str, ...]
    weight: float = 1.0

    def __post_init__(self) -> None:
        if (
            not self.name.strip()
            or not self.phrases
            or any(not value.strip() for value in self.phrases)
        ):
            raise ValueError("concept needs a name and nonempty lexical phrases")
        _check_weight(self.weight, f"concept {self.name!r} weight")


@dataclass(frozen=True)
class TargetSpec:
    """Lexical supported and unsupported concepts for one text unit."""

    sample_id: str
    text_unit: str
    positive: tuple[LexicalConcept, ...] = ()
    negative: tuple[LexicalConcept, ...] = ()
    sample_weight: float = 1.0
    provenance: str = ""

    def __post_init__(self) -> None:
        if not self.sample_id.strip() or not self.text_unit.strip():
            raise ValueError("sample_id and text_unit must be nonempty")
        _check_weight(self.sample_weight, "sample_weight")
        if not isinstance(self.provenance, str):
            raise TypeError("provenance must be a string")
        _check_unique_names(concept.name for concept in (*self.positive, *self.negative))


@dataclass(frozen=True)
class EncodingAudit:
    """Summary of an encoding run, saved alongside the encoded targets."""

    tokenizer_sha256: str
    sample_count: int
    positive_concepts: int
    negative_concepts: int
    overlap_token_ids: int


def tokenizer_fingerprint(tokenizer: Tokenizer) -> str:
    """SHA-256 of the token-to-ID mapping."""

    vocab = tokenizer.get_vocab()
    if not vocab:
        raise ValueError("tokenizer vocabulary is empty")
    if any(
        not isinstance(token, str)
        or isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        for token, token_id in vocab.items()
    ):
        raise ValueError("tokenizer vocabulary must map strings to nonnegative integer IDs")
    payload = json.dumps(dict(sorted(vocab.items())), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def encode_target_specs(
    specs: Sequence[TargetSpec], tokenizer: Tokenizer, *, allow_overlap: bool = False
) -> tuple[tuple[SampleTargets, ...], EncodingAudit]:
    """Encode each concept's phrases into the union of their token IDs.

    A token ID shared by the supported and unsupported sides of a row is an
    error unless ``allow_overlap=True``, in which case both sides keep it and
    the count is reported in the audit.
    """

    fingerprint = tokenizer_fingerprint(tokenizer)
    vocabulary_ids = set(tokenizer.get_vocab().values())
    special_ids = set(getattr(tokenizer, "all_special_ids", ()))
    seen: set[str] = set()
    rows: list[SampleTargets] = []
    overlaps = 0

    def encode_concept(concept: LexicalConcept, sample_id: str) -> ConceptTarget:
        found: set[int] = set()
        for phrase in concept.phrases:
            for token_id in tokenizer.encode(phrase, add_special_tokens=False):
                if isinstance(token_id, bool) or not isinstance(token_id, int):
                    raise ValueError(f"concept {concept.name!r} encodes to a noninteger ID")
                found.add(token_id)
        if not found:
            raise ValueError(f"concept {concept.name!r} encodes to no token IDs")
        target = ConceptTarget(concept.name, tuple(sorted(found)), concept.weight)
        _validate_concept_ids(target, vocabulary_ids, special_ids, sample_id)
        return target

    for spec in specs:
        if spec.sample_id in seen:
            raise ValueError(f"duplicate sample_id: {spec.sample_id!r}")
        seen.add(spec.sample_id)
        positive = tuple(encode_concept(concept, spec.sample_id) for concept in spec.positive)
        negative = tuple(encode_concept(concept, spec.sample_id) for concept in spec.negative)
        positive_ids = {token_id for concept in positive for token_id in concept.token_ids}
        negative_ids = {token_id for concept in negative for token_id in concept.token_ids}
        shared = positive_ids & negative_ids
        overlaps += len(shared)
        if shared and not allow_overlap:
            raise ValueError(
                f"sample {spec.sample_id!r}: {len(shared)} token IDs appear on both the "
                "supported and unsupported side; revise the lexical forms or pass "
                "allow_overlap=True"
            )
        rows.append(
            SampleTargets(
                spec.sample_id,
                positive,
                negative,
                spec.sample_weight,
                spec.text_unit,
                spec.provenance,
            )
        )

    audit = EncodingAudit(
        fingerprint,
        len(rows),
        sum(len(row.positive) for row in rows),
        sum(len(row.negative) for row in rows),
        overlaps,
    )
    return tuple(rows), audit


def save_target_specs(path: str | Path, specs: Sequence[TargetSpec]) -> None:
    """Write lexical target rows as JSONL, one row per line."""

    with Path(path).open("w", encoding="utf-8") as handle:
        for spec in specs:
            payload = {
                "schema_version": SPEC_VERSION,
                "sample_id": spec.sample_id,
                "text_unit": spec.text_unit,
                "sample_weight": spec.sample_weight,
                "provenance": spec.provenance,
                "positive_concepts": [_lexical_concept(concept) for concept in spec.positive],
                "negative_concepts": [_lexical_concept(concept) for concept in spec.negative],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_target_specs(path: str | Path) -> tuple[TargetSpec, ...]:
    """Read lexical target rows from JSONL."""

    rows: list[TargetSpec] = []
    for line_number, raw in _jsonl(path):
        try:
            if raw["schema_version"] != SPEC_VERSION:
                raise ValueError("unsupported schema_version")
            rows.append(
                TargetSpec(
                    sample_id=_require_string(raw["sample_id"], "sample_id"),
                    text_unit=_require_string(raw["text_unit"], "text_unit"),
                    positive=_parse_lexical(raw.get("positive_concepts", [])),
                    negative=_parse_lexical(raw.get("negative_concepts", [])),
                    sample_weight=float(raw.get("sample_weight", 1.0)),
                    provenance=raw.get("provenance", ""),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid target row at {path}:{line_number}: {exc}") from exc
    return tuple(rows)


def save_encoded_targets(
    path: str | Path, rows: Sequence[SampleTargets], audit: EncodingAudit
) -> None:
    """Write encoded rows as JSONL, each tagged with the tokenizer fingerprint."""

    if len(rows) != audit.sample_count:
        raise ValueError("audit sample count differs from encoded targets")
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: item.sample_id):
            payload = {
                "schema_version": ENCODED_VERSION,
                "tokenizer_sha256": audit.tokenizer_sha256,
                "sample_id": row.sample_id,
                "text_unit": row.text_unit,
                "sample_weight": row.sample_weight,
                "provenance": row.provenance,
                "positive_concepts": [_encoded_concept(concept) for concept in row.positive],
                "negative_concepts": [_encoded_concept(concept) for concept in row.negative],
            }
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def load_encoded_targets(path: str | Path, tokenizer: Tokenizer) -> tuple[SampleTargets, ...]:
    """Read encoded rows and check them against ``tokenizer``.

    Fails if the vocabulary fingerprint differs, or if any ID is missing from
    the vocabulary or is a special token.
    """

    fingerprint = tokenizer_fingerprint(tokenizer)
    vocabulary_ids = set(tokenizer.get_vocab().values())
    special_ids = set(getattr(tokenizer, "all_special_ids", ()))
    rows: list[SampleTargets] = []
    seen: set[str] = set()
    for line_number, raw in _jsonl(path):
        try:
            if raw["schema_version"] != ENCODED_VERSION:
                raise ValueError("unsupported schema_version")
            if raw["tokenizer_sha256"] != fingerprint:
                raise ValueError("tokenizer vocabulary fingerprint mismatch")
            row = SampleTargets(
                sample_id=_require_string(raw["sample_id"], "sample_id"),
                text_unit=_require_string(raw["text_unit"], "text_unit"),
                positive=_parse_encoded(raw.get("positive_concepts", [])),
                negative=_parse_encoded(raw.get("negative_concepts", [])),
                sample_weight=float(raw.get("sample_weight", 1.0)),
                provenance=raw.get("provenance", ""),
            )
            for concept in (*row.positive, *row.negative):
                _validate_concept_ids(concept, vocabulary_ids, special_ids, row.sample_id)
            if row.sample_id in seen:
                raise ValueError("duplicate sample_id")
            seen.add(row.sample_id)
            rows.append(row)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid encoded target at {path}:{line_number}: {exc}") from exc
    return tuple(rows)


class EncodedTargetStore:
    """Look up encoded targets by sample ID, in batch order."""

    def __init__(self, rows: Sequence[SampleTargets]) -> None:
        by_id: dict[str, SampleTargets] = {}
        for row in rows:
            if row.sample_id in by_id:
                raise ValueError(f"duplicate sample_id: {row.sample_id!r}")
            by_id[row.sample_id] = row
        self._by_id = by_id

    @classmethod
    def from_jsonl(cls, path: str | Path, tokenizer: Tokenizer) -> EncodedTargetStore:
        return cls(load_encoded_targets(path, tokenizer))

    def for_batch(self, sample_ids: Sequence[str]) -> tuple[SampleTargets, ...]:
        """Return one row per ID. A missing ID raises ``KeyError``."""

        try:
            return tuple(self._by_id[sample_id] for sample_id in sample_ids)
        except KeyError as exc:
            raise KeyError(f"no diagnostic target row for sample_id {exc.args[0]!r}") from exc

    def __len__(self) -> int:
        return len(self._by_id)


def _check_weight(value: float, field: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be finite and nonnegative")


def _check_unique_names(names: Iterable[str]) -> None:
    listed = list(names)
    if len(set(listed)) != len(listed):
        raise ValueError("concept names must be unique across both polarities")


def _validate_concept_ids(
    concept: ConceptTarget, vocabulary_ids: set[int], special_ids: set[int], sample_id: str
) -> None:
    context = f"sample {sample_id!r}, concept {concept.name!r}"
    missing = set(concept.token_ids) - vocabulary_ids
    if missing:
        raise ValueError(f"{context} has IDs absent from the vocabulary: {sorted(missing)}")
    special = special_ids.intersection(concept.token_ids)
    if special:
        raise ValueError(f"{context} contains special token IDs: {sorted(special)}")


def _jsonl(path: str | Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"invalid row at {path}:{line_number}: expected an object")
            rows.append((line_number, cast(dict[str, Any], raw)))
    return rows


def _parse_lexical(raw: Any) -> tuple[LexicalConcept, ...]:
    if not isinstance(raw, list):
        raise TypeError("concepts must be a list")
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("phrases"), list):
            raise TypeError("each concept needs a list of phrases")
    return tuple(
        LexicalConcept(
            name=_require_string(item["name"], "concept name"),
            phrases=tuple(_require_string(value, "phrase") for value in item["phrases"]),
            weight=float(item.get("weight", 1.0)),
        )
        for item in raw
    )


def _parse_encoded(raw: Any) -> tuple[ConceptTarget, ...]:
    if not isinstance(raw, list):
        raise TypeError("concepts must be a list")
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("token_ids"), list):
            raise TypeError("each encoded concept needs a list of token IDs")
    return tuple(
        ConceptTarget(
            name=_require_string(item["name"], "concept name"),
            token_ids=tuple(item["token_ids"]),
            weight=float(item.get("weight", 1.0)),
        )
        for item in raw
    )


def _lexical_concept(concept: LexicalConcept) -> dict[str, Any]:
    return {"name": concept.name, "phrases": list(concept.phrases), "weight": concept.weight}


def _encoded_concept(concept: ConceptTarget) -> dict[str, Any]:
    return {"name": concept.name, "token_ids": list(concept.token_ids), "weight": concept.weight}


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    return value
