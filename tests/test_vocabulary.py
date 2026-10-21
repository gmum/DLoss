import json
from pathlib import Path

import pytest

from diagnostic_loss import DiagnosticVocabulary, encode_target_specs

EXAMPLE = Path(__file__).parents[1] / "examples" / "example_vocabulary.json"


def vocabulary() -> DiagnosticVocabulary:
    return DiagnosticVocabulary(
        {
            "liver": {"normal": ("normal",), "cyst": ("cyst", "cysts"), "mass": ("mass",)},
            "kidneys": {"normal": ("normal",), "stone": ("stone",)},
        },
        name="test",
    )


def test_unsupported_defaults_to_other_concepts_of_the_same_organ():
    spec = vocabulary().target_spec("case::liver", "liver", ["cyst"])
    assert [c.name for c in spec.positive] == ["cyst"]
    assert [c.name for c in spec.negative] == ["normal", "mass"]
    assert spec.text_unit == "liver finding"
    assert spec.provenance == "vocabulary test"


def test_leading_space_variants():
    spec = vocabulary().target_spec("case::liver", "liver", ["cyst"])
    assert spec.positive[0].phrases == ("cyst", "cysts", " cyst", " cysts")
    plain = vocabulary().target_spec("case::liver", "liver", ["cyst"], add_leading_space=False)
    assert plain.positive[0].phrases == ("cyst", "cysts")


def test_explicit_unsupported_and_empty_rows():
    spec = vocabulary().target_spec("a", "liver", ["cyst"], unsupported=["mass"])
    assert [c.name for c in spec.negative] == ["mass"]
    empty = vocabulary().target_spec("b", "liver", [], unsupported=[])
    assert not empty.positive and not empty.negative
    with pytest.raises(ValueError, match="pass unsupported explicitly"):
        vocabulary().target_spec("c", "liver", [])


def test_rejects_unknown_and_contradictory_concepts():
    with pytest.raises(KeyError, match="unknown organ"):
        vocabulary().target_spec("a", "spleen", ["normal"])
    with pytest.raises(KeyError, match="unknown concept 'stone'"):
        vocabulary().target_spec("a", "liver", ["stone"])
    with pytest.raises(ValueError, match="both supported and unsupported"):
        vocabulary().target_spec("a", "liver", ["cyst"], unsupported=["cyst"])


def test_rejects_a_phrase_shared_by_two_concepts_of_one_organ():
    with pytest.raises(ValueError, match="listed under both"):
        DiagnosticVocabulary({"liver": {"cyst": ("cyst",), "lesion": ("Cyst",)}})


def test_from_json(tmp_path):
    path = tmp_path / "vocabulary.json"
    path.write_text(json.dumps({"name": "v", "organs": {"liver": {"cyst": ["cyst"]}}}))
    loaded = DiagnosticVocabulary.from_json(path)
    assert loaded.name == "v"
    assert loaded.concepts("liver") == ("cyst",)
    path.write_text(json.dumps({"organs": {"liver": {"cyst": "cyst"}}}))
    with pytest.raises(ValueError, match="list of strings"):
        DiagnosticVocabulary.from_json(path)


def test_example_vocabulary_has_no_token_collisions_with_qwen():
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    except OSError:
        pytest.skip("Qwen2.5 tokenizer not available offline")
    example = DiagnosticVocabulary.from_json(EXAMPLE)
    specs = [
        example.target_spec(f"{organ}::{concept}", organ, [concept])
        for organ in example.organs
        for concept in example.concepts(organ)
    ]
    _, audit = encode_target_specs(specs, tokenizer)
    assert audit.overlap_token_ids == 0
