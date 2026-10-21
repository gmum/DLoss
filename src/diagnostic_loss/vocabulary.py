"""Build target rows from a per-organ diagnostic vocabulary.

A vocabulary maps each organ to its diagnostic concepts and each concept to
its lexical forms. Given the concepts a finding supports, the remaining
concepts of the same organ become its unsupported alternatives.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .targets import LexicalConcept, TargetSpec


@dataclass(frozen=True)
class DiagnosticVocabulary:
    organs: Mapping[str, Mapping[str, tuple[str, ...]]]
    name: str = ""

    def __post_init__(self) -> None:
        if not self.organs:
            raise ValueError("vocabulary has no organs")
        for organ, concepts in self.organs.items():
            if not concepts:
                raise ValueError(f"organ {organ!r} has no concepts")
            owner: dict[str, str] = {}
            for concept, phrases in concepts.items():
                if not phrases or any(not phrase.strip() for phrase in phrases):
                    raise ValueError(f"{organ}/{concept} needs nonempty phrases")
                for phrase in phrases:
                    key = phrase.strip().lower()
                    if key in owner and owner[key] != concept:
                        raise ValueError(
                            f"phrase {phrase!r} is listed under both {organ}/{owner[key]} "
                            f"and {organ}/{concept}"
                        )
                    owner[key] = concept

    @classmethod
    def from_json(cls, path: str | Path) -> DiagnosticVocabulary:
        """Load ``{"name": ..., "organs": {organ: {concept: [phrase, ...]}}}``."""

        with Path(path).open(encoding="utf-8") as handle:
            raw: Any = json.load(handle)
        if not isinstance(raw, dict) or not isinstance(raw.get("organs"), dict):
            raise ValueError(f"{path}: expected an object with an 'organs' mapping")
        organs: dict[str, dict[str, tuple[str, ...]]] = {}
        for organ, concepts in raw["organs"].items():
            if not isinstance(concepts, dict):
                raise ValueError(f"{path}: organ {organ!r} must map concepts to phrase lists")
            organs[organ] = {}
            for concept, phrases in concepts.items():
                if not isinstance(phrases, list) or not all(isinstance(p, str) for p in phrases):
                    raise ValueError(f"{path}: {organ}/{concept} must be a list of strings")
                organs[organ][concept] = tuple(phrases)
        return cls(organs, str(raw.get("name", "")))

    def concepts(self, organ: str) -> tuple[str, ...]:
        return tuple(self._organ(organ))

    def target_spec(
        self,
        sample_id: str,
        organ: str,
        supported: Iterable[str],
        unsupported: Iterable[str] | None = None,
        *,
        add_leading_space: bool = True,
        sample_weight: float = 1.0,
    ) -> TargetSpec:
        """Target row for one organ finding.

        ``unsupported`` defaults to every concept of ``organ`` not in
        ``supported``. With ``add_leading_space``, each phrase is also added
        with a leading space, since most subword tokenizers encode a word
        differently at the start of the text and after a space.
        """

        concepts = self._organ(organ)
        positive = list(dict.fromkeys(supported))
        if unsupported is None:
            if not positive:
                raise ValueError(
                    "no supported concepts given; pass unsupported explicitly "
                    "(an empty list produces a row without supervision)"
                )
            negative = [name for name in concepts if name not in positive]
        else:
            negative = list(dict.fromkeys(unsupported))
        for name in (*positive, *negative):
            if name not in concepts:
                raise KeyError(f"unknown concept {name!r} for organ {organ!r}")
        both = set(positive) & set(negative)
        if both:
            raise ValueError(f"concepts cannot be both supported and unsupported: {sorted(both)}")

        def lexical(name: str) -> LexicalConcept:
            phrases = list(concepts[name])
            if add_leading_space:
                phrases += [" " + p for p in concepts[name] if not p.startswith(" ")]
            return LexicalConcept(name, tuple(dict.fromkeys(phrases)))

        return TargetSpec(
            sample_id=sample_id,
            text_unit=f"{organ} finding",
            positive=tuple(lexical(name) for name in positive),
            negative=tuple(lexical(name) for name in negative),
            sample_weight=sample_weight,
            provenance=f"vocabulary {self.name}" if self.name else "",
        )

    def _organ(self, organ: str) -> Mapping[str, tuple[str, ...]]:
        try:
            return self.organs[organ]
        except KeyError:
            raise KeyError(
                f"unknown organ {organ!r}; vocabulary has {sorted(self.organs)}"
            ) from None
