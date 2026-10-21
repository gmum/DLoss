import runpy
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parents[1] / "examples"


@pytest.mark.parametrize("script", ["quickstart.py", "custom_layout.py"])
def test_offline_example_runs(script, capsys):
    runpy.run_path(str(EXAMPLES / script), run_name="__main__")
    assert capsys.readouterr().out
