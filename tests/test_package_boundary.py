"""Build and use the wheel outside the source tree and both research checkouts."""

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path


def test_built_package_runs_public_workflow_outside_source_tree(tmp_path):
    import jax
    import numpy as np

    from ilg_toolkit import (
        FitConfig,
        PairwiseObservations,
        PreparedRegion,
        TargetSpec,
        fit,
        save_predictor,
    )
    from ilg_toolkit.models import UNetEmbeddingDistance

    inference_region = PreparedRegion(
        "independent",
        np.arange(32).reshape(4, 4, 2) / 32,
        ("a", "b"),
        np.array([[0, 0], [3, 3]]),
    )
    inference_observations = PairwiseObservations.from_matrix(
        ("a", "b"), [[0, 0.3], [0.3, 0]], target=TargetSpec("synthetic", units="index")
    )
    source_predictor = fit(
        inference_region,
        inference_observations,
        model=UNetEmbeddingDistance(
            2, patch_size=1, base_channels=2, embedding_dim=2, dropout=0.2, key=jax.random.key(1)
        ),
        config=FitConfig(epochs=0),
    ).predictor
    inference_artifact = tmp_path / "saved-predictor.ilg"
    save_predictor(inference_artifact, source_predictor)
    expected_inference = source_predictor.predict(inference_region).values.tolist()
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
reloaded = ilg.load_predictor({str(inference_artifact)!r})
np.testing.assert_allclose(reloaded.predict(region).values, {expected_inference!r},
    rtol=1e-6, atol=1e-7)
obs = ilg.PairwiseObservations.from_matrix(('a','b'), [[0,.3],[.3,0]],
    target=ilg.TargetSpec('synthetic', units='index'))
result = ilg.fit(region, obs, config=ilg.FitConfig(epochs=0))
assert np.isfinite(result.predictor.predict(region).values).all()
assert result.predictor.predict(region).target.units == 'index'
assert ilg.__file__.startswith({str(installation)!r})
import runpy
runpy.run_path({str(root / "examples" / "conductance.py")!r}, run_name='__main__')
print(ilg.__file__)
"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["JAX_PLATFORMS"] = "cpu"
    environment["JAX_ENABLE_X64"] = "true"
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
