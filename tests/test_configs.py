from pathlib import Path

import pytest

from diagnostic_loss import DiagnosticLossConfig

yaml = pytest.importorskip("yaml")

CONFIGS = Path(__file__).parents[1] / "configs" / "paper"

# Appendix table "Diagnostic Loss coefficients in the main experiments".
PAPER = {
    "merlin": (0.001, 0.0005),
    "fvlm": (0.001, 0.00025),
    "btb3d": (0.001, 0.00025),
    "hlip": (0.001, 0.00025),
    "radfm": (0.01, 0.01),
    "spectre": (0.01, 0.005),
}


@pytest.mark.parametrize("name", sorted(PAPER))
def test_paper_configs_match_the_reported_coefficients(name):
    config = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())
    loss = DiagnosticLossConfig(**config["diagnostic_loss"])
    assert (loss.lambda_pos, loss.lambda_neg) == PAPER[name]
    assert loss.tau == 8.0


def test_every_config_is_checked():
    assert {path.stem for path in CONFIGS.glob("*.yaml")} == set(PAPER)
