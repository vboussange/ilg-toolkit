"""Native group identity and coordinated holdout contracts through public APIs."""

from dataclasses import replace

import numpy as np
import pytest
from test_species_mlpe import irregular_problem

from ilg_toolkit import (
    ObservationPartition,
    PairwiseObservations,
    PopulationFold,
    generate_population_folds,
)


def test_nested_holdouts_are_deterministic_independent_and_have_no_validation():
    regions, observations, _ = irregular_problem()
    # Use complete observed edges so every random draw has train and query pairs.
    for region in regions:
        for species, measured in observations[region.name].items():
            labels = region.sampling_units[species].ids
            pairs = [(a, b) for i, a in enumerate(labels) for b in labels[i + 1 :]]
            observations[region.name][species] = PairwiseObservations.from_pairs(
                pairs, np.arange(len(pairs)) + 1.0, target=measured.target
            )
    sizes = {r.name: {s: 2 for s in observations[r.name]} for r in regions}
    first = generate_population_folds(regions, observations, n_folds=3, holdout_size=sizes, seed=23)
    reordered = {
        name: {
            species: PairwiseObservations.from_pairs(
                [(b, a) for a, b in measured.observed_pairs[::-1]],
                measured.observed_values[::-1],
                target=measured.target,
                sampling_unit_ids=measured.sampling_unit_ids[::-1],
            )
            for species, measured in reversed(list(groups.items()))
        }
        for name, groups in reversed(list(observations.items()))
    }
    reversed_regions = [
        replace(
            r,
            sampling_units={
                s: replace(u, ids=u.ids[::-1], grid_positions=np.asarray(u.grid_positions)[::-1])
                for s, u in reversed(list(r.sampling_units.items()))
            },
        )
        for r in regions[::-1]
    ]
    assert first == generate_population_folds(
        reversed_regions, reordered, n_folds=3, holdout_size=sizes, seed=23
    )
    assert all(f.validation == {} for f in first)
    for fold in first:
        for region, groups in fold.held_out_units.items():
            for species, held in groups.items():
                assert len(held) == 2
                assert all(not set(p) & set(held) for p in fold.training[region][species].pairs)
                assert all(set(p) <= set(held) for p in fold.query[region][species].pairs)
    assert any(
        f.held_out_units["alpine"]["fox"] != f.held_out_units["alpine"]["lynx"] for f in first
    )


def test_fold_rejects_wrong_group_identity_and_leaking_training_endpoints():
    train = ObservationPartition("alpine", "fox", (("a", "b"),), "training")
    held = {"alpine": {"fox": ("c", "d")}}
    for invalid in [
        replace(train, species_name="lynx"),
        replace(train, region_name="valley"),
        replace(train, role="query"),
    ]:
        with pytest.raises(ValueError, match="identit|matching"):
            PopulationFold("f", held, {"alpine": {"fox": invalid}})
    with pytest.raises(ValueError, match="held-out endpoints"):
        PopulationFold("f", held, {"alpine": {"fox": replace(train, pairs=(("a", "c"),))}})
    with pytest.raises(ValueError, match="match exactly"):
        PopulationFold(
            "f", held, {"alpine": {"fox": train, "lynx": replace(train, species_name="lynx")}}
        )


def test_native_members_match_independent_fits_with_explicit_keys_and_final_selection():
    import equinox as eqx
    import jax
    from test_checkpoint import assert_same_state
    from test_mlpe_training import WeightedEmbedding, embedding_problem

    from ilg_toolkit import TrainingConfig, fit, fit_ensemble

    region, observed = embedding_problem()
    region = replace(
        region,
        sampling_units={
            "fox": region.sampling_units["species"],
            "lynx": region.sampling_units["species"],
        },
    )
    observations = {region.name: {"fox": observed, "lynx": observed}}
    folds = generate_population_folds([region], observations, n_folds=1, holdout_size=2, seed=47)

    def factory(key):
        return WeightedEmbedding(jax.random.normal(key, (2,)) * 0.1, species=("fox", "lynx"))

    config = TrainingConfig(objective="mlpe", steps=2, evaluation_interval=2)
    ensemble = fit_ensemble(
        [region],
        observations,
        folds=folds,
        initialization_seeds=(13, 29),
        config=config,
        model_factory=factory,
    )
    for member in ensemble.members:
        assert member.status == "completed", member.failure
        reference = fit(
            [region],
            observations,
            distance_model=factory(jax.random.PRNGKey(member.identity.initialization_seed)),
            config=replace(config, seed=member.identity.initialization_seed),
            partition=folds[0].training,
        )
        assert member.training_state is not None
        assert_same_state(member.training_state, reference)
        assert member.training_state.selection == "final"
        assert member.training_state.selected_step == 2
        assert eqx.tree_equal(member.model, reference.model)


def test_selected_species_deployment_preserves_pair_iterables_and_original_scale():
    import equinox as eqx
    from ensemble_helpers import labelled_ensemble

    from ilg_toolkit import Ensemble

    region, ensemble, target = labelled_ensemble()
    target = replace(target, transform="log1p")
    members = tuple(
        replace(member, model=eqx.tree_at(lambda m: m.target, member.model, target))
        for member in ensemble.members
    )
    ensemble = Ensemble(members)
    with pytest.raises(ValueError, match="species"):
        ensemble.predict(region)
    selected = ensemble.predict(
        region,
        species=("lynx", "fox"),
        pairs={name: iter([("d", "c"), ("a", "c")]) for name in ("fox", "lynx")},
    )
    assert tuple(selected) == ("lynx", "fox")
    for name, distances in [
        ("fox", np.array([[1, 4], [4, 16]])),
        ("lynx", np.array([[9, 36], [16, 64]])),
    ]:
        expected = np.expm1(distances)
        np.testing.assert_allclose(selected[name].member_values, expected, rtol=1e-12)
        np.testing.assert_allclose(
            selected[name].prediction.values, expected.mean(axis=0), rtol=1e-12
        )
        assert selected[name].prediction.species_name == name
        assert selected[name].prediction.pairs == (("d", "c"), ("a", "c"))
    assert set(ensemble.predict(region, species="fox")) == {"fox"}


@pytest.mark.parametrize("objective", ["direct_log1p", "mlpe"])
def test_loglinear_native_members_fit_and_predict_each_species(objective):
    import equinox as eqx
    import jax
    from test_mlpe_training import embedding_problem

    from ilg_toolkit import TrainingConfig, UnpooledMLPECalibrationModule, fit_ensemble
    from ilg_toolkit.models import LogLinearConductance

    region, observed = embedding_problem()
    region = replace(
        region, sampling_units={name: region.sampling_units["species"] for name in ("fox", "lynx")}
    )
    other = PairwiseObservations.from_pairs(
        observed.observed_pairs, 1 + 2 * observed.observed_values, target=observed.target
    )
    observations = {region.name: {"fox": observed, "lynx": other}}

    def factory(key):
        model = LogLinearConductance(2, species=("fox", "lynx"), patch_size=1)
        return eqx.tree_at(lambda m: m.coefficients, model, jax.random.normal(key, (2, 2)) * 0.05)

    ensemble = fit_ensemble(
        [region],
        observations,
        n_folds=1,
        holdout_size=2,
        fold_seed=47,
        config=TrainingConfig(objective=objective, steps=1),
        model_factory=factory,
    )
    member = ensemble.members[0]
    assert member.status == "completed", member.failure
    assert member.model is not None and member.training_state is not None
    assert isinstance(member.model.distance_model, LogLinearConductance)
    assert member.training_state.step == member.training_state.selected_step == 1
    queries = {name: partition.pairs for name, partition in member.fold.query[region.name].items()}
    prediction = ensemble.predict(region, species=("fox", "lynx"), pairs=queries)
    surfaces = ensemble.conductance_surfaces(region, species=("fox", "lynx"))
    matrices = ensemble.ecological_distances(region, species=("fox", "lynx"))
    assert set(prediction) == set(surfaces) == set(matrices) == {"fox", "lynx"}
    for name in prediction:
        assert prediction[name].prediction.species_name == surfaces[name].species_name == name
        assert prediction[name].prediction.pairs == queries[name]
        assert np.isfinite(prediction[name].prediction.values).all()
        assert set(matrices[name]) == {member.identity.member_id}
    if objective == "mlpe":
        assert isinstance(member.model.calibration, UnpooledMLPECalibrationModule)
        assert set(member.model.calibration.groups[region.name]) == {"fox", "lynx"}


def test_native_registry_identity_and_nested_fold_inputs_reject_mismatches():
    from ensemble_helpers import labelled_ensemble
    from test_ensemble import model_factory
    from test_mlpe_training import embedding_problem

    from ilg_toolkit import Ensemble, TrainingConfig, fit_ensemble

    region, ensemble, _ = labelled_ensemble()
    first, second = ensemble.members
    assert second.model is not None
    changed = replace(
        second.model, distance_model=replace(second.model.distance_model, species=("lynx", "fox"))
    )
    with pytest.raises(ValueError, match="ordered species"):
        Ensemble((first, replace(second, model=changed)))
    with pytest.raises(ValueError, match="species"):
        ensemble.predict(region, species="unknown")
    region, observed = embedding_problem()
    observations = {region.name: {"species": observed}}
    for size in [{region.name: {}}, {region.name: {"other": 2}}, {"unknown": {"species": 2}}]:
        with pytest.raises(ValueError, match="holdout_size"):
            generate_population_folds([region], observations, n_folds=1, holdout_size=size)
    fold = generate_population_folds([region], observations, n_folds=1, holdout_size=2)[0]
    with pytest.raises(ValueError, match="prepared region/species"):
        fit_ensemble(
            [region],
            observations,
            folds=[
                replace(fold, held_out_units={region.name: {"species": ("missing",)}}, query={})
            ],
            config=TrainingConfig(objective="direct_log1p", steps=0),
            model_factory=model_factory,
        )


def test_automatic_sampling_unit_holdouts_support_direct_individual_observations():
    import jax.numpy as jnp
    from test_mlpe_training import WeightedEmbedding, embedding_problem

    from ilg_toolkit import TrainingConfig, fit_ensemble

    region, observed = embedding_problem()
    region = replace(
        region,
        sampling_units={
            "species": replace(region.sampling_units["species"], kinds=("individual",) * 6)
        },
    )
    ensemble = fit_ensemble(
        [region],
        {region.name: {"species": observed}},
        n_folds=1,
        holdout_size=2,
        config=TrainingConfig(objective="direct_log1p", steps=0),
        model_factory=lambda key: WeightedEmbedding(jnp.zeros(2)),
    )
    assert ensemble.members[0].status == "completed"
