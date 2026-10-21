import json

import pytest

from diagnostic_loss import (
    EncodedTargetStore,
    LexicalConcept,
    TargetSpec,
    encode_target_specs,
    load_encoded_targets,
    load_target_specs,
    save_encoded_targets,
    save_target_specs,
)


class ToyTokenizer:
    all_special_ids = (0,)

    def __init__(self, changed=False):
        self.vocab = {"<pad>": 0, "cyst": 1, "stone": 2, "finding": 3}
        if changed:
            self.vocab["cyst"], self.vocab["stone"] = 2, 1

    def encode(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        return [self.vocab[word] for word in text.split()]

    def get_vocab(self):
        return self.vocab


def test_encode_save_load_and_fingerprint(tmp_path):
    specs = [
        TargetSpec(
            "case::organ",
            "organ finding",
            (LexicalConcept("supported", ("cyst",)),),
            (LexicalConcept("unsupported", ("stone",)),),
            provenance="synthetic manual tags; unit sample and concept weights",
        )
    ]
    encoded, audit = encode_target_specs(specs, ToyTokenizer())
    assert audit.positive_concepts == audit.negative_concepts == 1
    assert audit.overlap_token_ids == 0
    assert encoded[0].provenance == specs[0].provenance
    path = tmp_path / "encoded.jsonl"
    save_encoded_targets(path, encoded, audit)
    assert load_encoded_targets(path, ToyTokenizer()) == encoded
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_encoded_targets(path, ToyTokenizer(changed=True))


def test_encoded_target_store_selects_batch_order_and_rejects_missing_rows(tmp_path):
    specs = [
        TargetSpec("one", "liver", (LexicalConcept("cyst", ("cyst",)),)),
        TargetSpec("empty", "kidney"),
    ]
    rows, audit = encode_target_specs(specs, ToyTokenizer())
    path = tmp_path / "encoded.jsonl"
    save_encoded_targets(path, rows, audit)
    store = EncodedTargetStore.from_jsonl(path, ToyTokenizer())
    assert len(store) == 2
    selected = store.for_batch(["empty", "one", "one"])
    assert [row.sample_id for row in selected] == ["empty", "one", "one"]
    assert not selected[0].has_targets
    assert selected[1] is selected[2]
    with pytest.raises(KeyError, match="no diagnostic target row for sample_id 'missing'"):
        store.for_batch(["one", "missing"])
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        EncodedTargetStore.from_jsonl(path, ToyTokenizer(changed=True))
    with pytest.raises(ValueError, match="duplicate sample_id"):
        EncodedTargetStore([rows[0], rows[0]])


def test_collision_is_visible_and_not_silently_changed(tmp_path):
    spec = TargetSpec(
        "one", "liver", (LexicalConcept("a", ("cyst",)),), (LexicalConcept("b", ("cyst",)),)
    )
    with pytest.raises(ValueError, match="supported and unsupported side"):
        encode_target_specs([spec], ToyTokenizer())
    encoded, audit = encode_target_specs([spec], ToyTokenizer(), allow_overlap=True)
    assert audit.overlap_token_ids == 1
    assert encoded[0].positive[0].token_ids == encoded[0].negative[0].token_ids
    path = tmp_path / "overlap.jsonl"
    save_encoded_targets(path, encoded, audit)
    assert load_encoded_targets(path, ToyTokenizer()) == encoded


def test_reject_special_and_empty_phrases():
    with pytest.raises(ValueError, match="special token"):
        encode_target_specs(
            [TargetSpec("one", "liver", (LexicalConcept("bad", ("<pad>",)),))], ToyTokenizer()
        )
    with pytest.raises(ValueError, match="nonempty lexical"):
        LexicalConcept("bad", ("",))


def test_load_versioned_spec(tmp_path):
    path = tmp_path / "targets.jsonl"
    path.write_text(
        json.dumps(
            {
                "schema_version": "diagnostic-loss-spec-v1",
                "sample_id": "one",
                "text_unit": "liver",
                "positive_concepts": [{"name": "cyst", "phrases": ["cyst"]}],
                "negative_concepts": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    specs = load_target_specs(path)
    assert specs[0].positive[0].phrases == ("cyst",)


def test_reject_malformed_phrase_field(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(
            {
                "schema_version": "diagnostic-loss-spec-v1",
                "sample_id": "one",
                "text_unit": "liver",
                "positive_concepts": [{"name": "bad", "phrases": "cyst"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="list of phrases"):
        load_target_specs(path)


@pytest.mark.parametrize("field", ["positive_concepts", "negative_concepts"])
@pytest.mark.parametrize("token_id, message", [(0, "special token"), (2, "absent"), (99, "absent")])
def test_load_rejects_invalid_ids_with_sample_and_concept_context(
    tmp_path, field, token_id, message
):
    tokenizer = ToyTokenizer()
    tokenizer.vocab = {"<pad>": 0, "cyst": 1, "stone": 4}
    rows, audit = encode_target_specs(
        [TargetSpec("case::liver", "liver", (LexicalConcept("cyst", ("cyst",)),))], tokenizer
    )
    path = tmp_path / "targets.jsonl"
    save_encoded_targets(path, rows, audit)
    raw = json.loads(path.read_text())
    raw["positive_concepts"] = []
    raw[field] = [{"name": "invalid-concept", "token_ids": [token_id], "weight": 0.0}]
    path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message) as error:
        EncodedTargetStore.from_jsonl(path, tokenizer)
    assert "case::liver" in str(error.value)
    assert "invalid-concept" in str(error.value)
    assert f"{path}:1" in str(error.value)


def test_load_checks_current_special_ids_even_when_vocabulary_matches(tmp_path):
    tokenizer = ToyTokenizer()
    rows, audit = encode_target_specs(
        [TargetSpec("one", "liver", (LexicalConcept("cyst", ("cyst",)),))], tokenizer
    )
    path = tmp_path / "targets.jsonl"
    save_encoded_targets(path, rows, audit)
    tokenizer.all_special_ids = (0, 1)
    with pytest.raises(ValueError, match="special token"):
        load_encoded_targets(path, tokenizer)


def test_explicit_boundary_variants_cover_contextual_tokens(tmp_path):
    class BoundaryTokenizer:
        all_special_ids = (0,)

        def get_vocab(self):
            return {"<pad>": 0, "c": 1, "yst": 2, " cyst": 3, "A": 4}

        def encode(self, text, *, add_special_tokens=False):
            assert not add_special_tokens
            return {"cyst": [1, 2], " cyst": [3], "A cyst": [4, 3]}[text]

    tokenizer = BoundaryTokenizer()
    path = tmp_path / "spec.jsonl"
    payload = {
        "schema_version": "diagnostic-loss-spec-v1",
        "sample_id": "one",
        "text_unit": "liver",
        "positive_concepts": [{"name": "cyst", "phrases": ["cyst"]}],
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    bare, _ = encode_target_specs(load_target_specs(path), tokenizer)
    contextual_ids = set(tokenizer.encode("A cyst")[1:])
    assert not contextual_ids.intersection(bare[0].positive[0].token_ids)

    payload["positive_concepts"][0]["phrases"].append(" cyst")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    specs = load_target_specs(path)
    assert specs[0].positive[0].phrases == ("cyst", " cyst")
    rows, audit = encode_target_specs(specs, tokenizer)
    assert rows[0].positive[0].token_ids == (1, 2, 3)
    assert contextual_ids.issubset(rows[0].positive[0].token_ids)
    encoded_path = tmp_path / "encoded.jsonl"
    save_encoded_targets(encoded_path, rows, audit)
    assert load_encoded_targets(encoded_path, tokenizer) == rows


@pytest.mark.parametrize(
    ("encoded", "field"),
    [
        (encoded, field)
        for encoded in (False, True)
        for field in ("sample_id", "text_unit", "name", "phrase")
        if not (encoded and field == "phrase")
    ],
)
@pytest.mark.parametrize("invalid", [None, 17])
def test_json_text_fields_reject_nonstrings(tmp_path, encoded, field, invalid):
    path = tmp_path / "targets.jsonl"
    specs = [TargetSpec("case", "finding", (LexicalConcept("cyst", ("cyst",)),))]
    if encoded:
        rows, audit = encode_target_specs(specs, ToyTokenizer())
        save_encoded_targets(path, rows, audit)
        payload = json.loads(path.read_text())
    else:
        payload = {
            "schema_version": "diagnostic-loss-spec-v1",
            "sample_id": "case",
            "text_unit": "finding",
            "positive_concepts": [{"name": "cyst", "phrases": ["cyst"]}],
        }
    if field in ("sample_id", "text_unit"):
        payload[field] = invalid
    elif field == "name":
        payload["positive_concepts"][0]["name"] = invalid
    else:
        payload["positive_concepts"][0]["phrases"] = [invalid]
    path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match=r"targets.jsonl:1: .*must be a string"):
        if encoded:
            load_encoded_targets(path, ToyTokenizer())
        else:
            load_target_specs(path)


def test_lexical_rows_round_trip_through_jsonl(tmp_path):
    specs = [
        TargetSpec(
            "case::liver",
            "liver finding",
            (LexicalConcept("cyst", ("cyst", " cyst"), 2.0),),
            (LexicalConcept("stone", ("stone",)),),
            sample_weight=0.5,
            provenance="manual",
        ),
        TargetSpec("case::kidney", "kidney finding"),
    ]
    path = tmp_path / "targets.jsonl"
    save_target_specs(path, specs)
    assert load_target_specs(path) == tuple(specs)
