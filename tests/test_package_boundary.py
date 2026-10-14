"""Build and use the wheel outside the source tree and both research checkouts."""

import json
import os
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import coordax as cx
import pytest


def test_built_package_runs_public_workflow_outside_source_tree(tmp_path):
    import jax
    import numpy as np

    from ilg_toolkit import (
        PairwiseObservations,
        RegionBatch,
        SamplingUnits,
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
        {"fox": SamplingUnits(("a", "b"), np.array([[0, 0], [3, 3]]))},
        feature_names=("elevation", "habitat"),
    )
    inference_observations = PairwiseObservations.from_matrix(
        ("a", "b"), [[0, 0.3], [0.3, 0]], target=TargetSpec("synthetic", units="index")
    )
    source_model = fit(
        [inference_region],
        {inference_region.name: {"fox": inference_observations}},
        distance_model=UNetEmbeddingDistance(
            2,
            species=("fox",),
            patch_size=1,
            base_channels=2,
            embedding_dim=2,
            dropout=0.2,
            key=jax.random.key(1),
        ),
        config=TrainingConfig(objective="direct_log1p", steps=0),
    ).model
    inference_artifact = tmp_path / "saved-model.ilg"
    save_model(inference_artifact, source_model)
    expected_inference = source_model.predict(inference_region)["fox"].values.tolist()
    checkpoint_artifact = tmp_path / "saved-checkpoint.ilg"

    class Interrupted(Exception):
        pass

    def save_and_interrupt(state):
        assert state.step == 1 and [record.step for record in state.history] == [0]
        save_checkpoint(checkpoint_artifact, state)
        raise Interrupted

    config = TrainingConfig(objective="direct_log1p", steps=2, evaluation_interval=2, seed=17)
    with jax.enable_x64():
        continued_source = fit(
            [inference_region],
            {inference_region.name: {"fox": inference_observations}},
            distance_model=source_model.distance_model,
            config=config,
        )
        with pytest.raises(Interrupted):
            fit(
                [inference_region],
                {inference_region.name: {"fox": inference_observations}},
                distance_model=source_model.distance_model,
                config=config,
                on_step=save_and_interrupt,
            )
        expected_continued = continued_source.model.predict(inference_region)["fox"].values.tolist()
        expected_history = [record.step for record in continued_source.history]
        assert expected_history == [0, 2] and continued_source.step == 2
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
    {{'fox': ilg.SamplingUnits(('a','b'), np.array([[0,0],[3,3]]))}},
    feature_names=('elevation','habitat'))
encoder = UNetEmbeddingDistance(2, species=("fox",), patch_size=1, base_channels=2, embedding_dim=2,
    dropout=0.2, key=jax.random.key(0))
reloaded = ilg.load_model({str(inference_artifact)!r}, distance_model=encoder)
np.testing.assert_allclose(reloaded.predict(region)["fox"].values, {expected_inference!r},
    rtol=1e-6, atol=1e-7)
obs = ilg.PairwiseObservations.from_matrix(('a','b'), [[0,.3],[.3,0]],
    target=ilg.TargetSpec('synthetic', units='index'))
state = ilg.load_checkpoint({str(checkpoint_artifact)!r}, distance_model=encoder)
assert state.step == 1 and state.selected_step == 0
assert state.config.steps == 2 and [record.step for record in state.history] == [0]
continued = ilg.fit([region], {{region.name: {{"fox": obs}}}}, state=state)
assert continued.step == 2 and continued.selected_step == 2
assert [record.step for record in continued.history] == {expected_history!r}
np.testing.assert_allclose(continued.model.predict(region)["fox"].values, {expected_continued!r},
    rtol=1e-6, atol=1e-7)
result = ilg.fit([region], {{region.name: {{"fox": obs}}}},
    distance_model=UNetEmbeddingDistance(2, species=("fox",), patch_size=1,
        base_channels=2, embedding_dim=2, key=jax.random.key(2)),
    config=ilg.TrainingConfig(objective="direct_log1p", steps=0))
assert np.isfinite(result.model.predict(region)["fox"].values).all()
assert result.model.predict(region)["fox"].target.units == 'index'
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
