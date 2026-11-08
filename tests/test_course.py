"""Every course module is a self-checking script: if all its asserts pass, the module passes."""
import runpy

import pytest

MODULES = [
    "m01_ndarray", "m02_broadcasting", "m03_linalg", "m04_random", "m05_einsum",
    "m06_memory", "m07_performance", "m08_numerics", "m09_extending", "m10_capstone",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_runs_clean(name, capsys):
    runpy.run_module(f"numpy_course.{name}", run_name="__main__")
    assert "complete: all assertions passed" in capsys.readouterr().out
