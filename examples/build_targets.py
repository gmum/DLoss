"""Turn labelled organ findings into target files for a given tokenizer.

Input is JSONL with one organ finding per line:

    {"sample_id": "case-1::liver", "organ": "liver", "supported": ["steatosis", "cyst"]}

An optional "unsupported" list overrides the default (every other concept of
the organ). Writes the lexical rows and the tokenizer-specific encoded rows.

    python examples/build_targets.py \\
        --findings examples/example_findings.jsonl \\
        --vocabulary examples/example_vocabulary.json \\
        --tokenizer Qwen/Qwen2.5-0.5B \\
        --out-dir targets/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from diagnostic_loss import (
    DiagnosticVocabulary,
    encode_target_specs,
    save_encoded_targets,
    save_target_specs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--findings", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True, help="Hugging Face name or local path")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    vocabulary = DiagnosticVocabulary.from_json(args.vocabulary)
    specs = []
    with args.findings.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                specs.append(
                    vocabulary.target_spec(
                        row["sample_id"], row["organ"], row["supported"], row.get("unsupported")
                    )
                )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    rows, audit = encode_target_specs(specs, tokenizer)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_target_specs(args.out_dir / "targets.jsonl", specs)
    save_encoded_targets(args.out_dir / "encoded_targets.jsonl", rows, audit)
    print(
        f"{audit.sample_count} rows, {audit.positive_concepts} supported and "
        f"{audit.negative_concepts} unsupported concepts -> {args.out_dir}"
    )


if __name__ == "__main__":
    main()
