"""Build and use the wheel outside the source tree and both research checkouts."""

import json
import os
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import coordax as cx


def test_built_package_runs_public_workflow_outside_source_tree(tmp_path):
    import jax
    import numpy as np

    from ilg_toolkit import (
        PairwiseObservations,
        RegionBatch,
        TargetSpec,
        TrainingConfig,
        fit,
        save_checkpoint,
        save_model,
    )
    from ilg_toolkit.models import UNetEmbeddingDistance

    inference_region = RegionBatch(
        "independent",
        cx.field(np.arange(32, dtype=np.float32).reshape(4, 4, 2) / 32, "row", "column", "feature"),
        ("a", "b"),
        np.array([[0, 0], [3, 3]]),
    )
    inference_observations = PairwiseObservations.from_matrix(
        ("a", "b"), [[0, 0.3], [0.3, 0]], target=TargetSpec("synthetic", units="index")
    )
    source_model = fit(
        inference_region,
        inference_observations,
        distance_model=UNetEmbeddingDistance(
            2, patch_size=1, base_channels=2, embedding_dim=2, dropout=0.2, key=jax.random.key(1)
        ),
        config=TrainingConfig(objective="direct_log1p", epochs=0),
    ).model
    inference_artifact = tmp_path / "saved-model.ilg"
    save_model(inference_artifact, source_model)
    expected_inference = source_model.predict(inference_region).values.tolist()
    checkpoint_artifact = tmp_path / "saved-checkpoint.ilg"
    completed = []
    with jax.enable_x64():
        continued_source = fit(
            inference_region,
            inference_observations,
            distance_model=source_model.distance_model,
            config=TrainingConfig(objective="direct_log1p", epochs=1),
            on_epoch=completed.append,
        )
        save_checkpoint(checkpoint_artifact, completed[0])
        expected_continued = continued_source.model.predict(inference_region).values.tolist()
    root = Path(__file__).resolve().parents[1]
    expected_version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
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
        assert not any(
            name.startswith(("experiments/", "ilg_toolkit/experimental/"))
            for name in archive.namelist()
        ), "Research diagnostics must not be installed with the toolkit"
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
import importlib, importlib.metadata, importlib.util, pkgutil
import numpy as np
import coordax as cx
import jax
import ilg_toolkit as ilg
from ilg_toolkit.models import UNetEmbeddingDistance
for module in pkgutil.walk_packages(ilg.__path__, ilg.__name__ + '.'):
    importlib.import_module(module.name)
assert importlib.metadata.version('ilg-toolkit') == {expected_version!r}
assert importlib.util.find_spec('ilg_toolkit.experimental') is None
assert not hasattr(ilg, 'diagnose_wishart')
assert importlib.util.find_spec('ilg_toolkit._archive') is None
assert importlib.util.find_spec('ilg_toolkit._codecs') is None
assert not any(name == 'deepilg' or name.startswith('deepilg.') for name in sys.modules)
region = ilg.RegionBatch('independent',
    cx.field(np.arange(32, dtype=np.float32).reshape(4,4,2)/32, 'row', 'column', 'feature'),
    ('a','b'), np.array([[0,0],[3,3]]))
encoder = UNetEmbeddingDistance(2, patch_size=1, base_channels=2, embedding_dim=2,
    dropout=0.2, key=jax.random.key(0))
reloaded = ilg.load_model({str(inference_artifact)!r}, distance_model=encoder)
np.testing.assert_allclose(reloaded.predict(region).values, {expected_inference!r},
    rtol=1e-6, atol=1e-7)
obs = ilg.PairwiseObservations.from_matrix(('a','b'), [[0,.3],[.3,0]],
    target=ilg.TargetSpec('synthetic', units='index'))
state = ilg.load_checkpoint({str(checkpoint_artifact)!r}, distance_model=encoder)
continued = ilg.fit(region, obs, state=state)
assert continued.epoch == 1
np.testing.assert_allclose(continued.model.predict(region).values, {expected_continued!r},
    rtol=1e-6, atol=1e-7)
result = ilg.fit(region, obs,
    distance_model=UNetEmbeddingDistance(2, patch_size=1, base_channels=2, embedding_dim=2,
        key=jax.random.key(2)),
    config=ilg.TrainingConfig(objective="direct_log1p", epochs=0))
assert np.isfinite(result.model.predict(region).values).all()
assert result.model.predict(region).target.units == 'index'
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
