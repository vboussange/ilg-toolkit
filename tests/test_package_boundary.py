"""Build and use the wheel outside the source tree and both research checkouts."""

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path


def test_built_package_runs_public_workflow_outside_source_tree(tmp_path):
    root = Path(__file__).resolve().parents[1]
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    built = subprocess.run(
        [
            sys.executable,
            "-c",
            "import setuptools.build_meta as b; b.build_wheel(" + repr(str(wheels)) + ")",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert built.returncode == 0, built.stderr
    installation = tmp_path / "installed"
    with zipfile.ZipFile(next(wheels.glob("*.whl"))) as archive:
        archive.extractall(installation)
    # Retain only installed third-party libraries from this interpreter. The
    # subprocess is isolated, runs elsewhere, and receives the built package.
    library_paths = [str(installation)] + [
        path
        for path in sys.path
        if path and "site-packages" in path and not Path(path).resolve().is_relative_to(root)
    ]
    script = f"""
import sys
sys.path[:0] = {json.dumps(library_paths)}
import importlib, importlib.metadata, pkgutil
import numpy as np
import ilg_toolkit as ilg
for module in pkgutil.walk_packages(ilg.__path__, ilg.__name__ + '.'):
    importlib.import_module(module.name)
assert importlib.metadata.version('ilg-toolkit') == '0.1.0'
assert not any(name == 'deepilg' or name.startswith('deepilg.') for name in sys.modules)
region = ilg.PreparedRegion('independent', np.arange(32).reshape(4,4,2)/32,
    ('a','b'), np.array([[0,0],[3,3]]))
obs = ilg.PairwiseObservations.from_matrix(('a','b'), [[0,.3],[.3,0]],
    target=ilg.TargetSpec('synthetic', units='index'))
result = ilg.fit(region, obs, config=ilg.FitConfig(epochs=0))
assert np.isfinite(result.predictor.predict(region).values).all()
assert result.predictor.predict(region).target.units == 'index'
assert ilg.__file__.startswith({str(installation)!r})
print(ilg.__file__)
"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
