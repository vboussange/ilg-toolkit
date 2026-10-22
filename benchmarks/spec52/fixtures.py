"""Revision-independent synthetic inputs for the frozen spec 52 protocol."""

import hashlib
import json

import numpy as np


def protocol():
    """Return the workload and numerical settings for the CPU comparison."""
    settings = {
        "protocol_version": 1,
        "frozen_main_commit": "8439f99ea37f04273a1868b8009e1076239ad316",
        "data_seed": 52053,
        "model_seed": 6,
        "training_seed": 7,
        "steps": 3,
        "warm_fit_calls": 1,
        "fit_repeats": 3,
        "prediction_repeats": 5,
        "learning_rate": 0.0001,
        "dtype": {
            "features": "float32",
            "neural_parameters": "float32",
            "targets": "float64",
            "resistance_and_mlpe": "float64",
        },
        "solver": {"rtol": 1e-08, "atol": 1e-10, "max_steps": 1000, "use_amg": False},
        "mlpe": {"variance_floor": 1e-10, "jitter": 0.0, "initial_variances": [0.1, 0.2]},
        "resnet": {"min_conductance": 1e-06, "patch_batch_size": 16},
        "unet": {"base_channels": 8, "embedding_dim": 4, "dropout": 0.1},
        "execution": {
            "backend": "cpu",
            "affinity_cpu_count": 2,
            "native_threads": 1,
            "persistent_compilation_cache": False,
            "xla_preallocate": False,
            "cpu_affinity": [0, 1],
        },
        "thresholds": {
            "all_cases_required": True,
            "all_outputs_finite": True,
            "raw_nonnegative_atol": 1e-07,
            "raw_symmetry_atol": 1e-06,
            "raw_diagonal_atol": 1e-06,
            "raw_mlpe_min_standard_deviation": 1e-08,
            "dense_resistance_rtol": 1e-05,
            "dense_resistance_atol": 1e-07,
            "warm_fit_median_ratio_pr_over_main_max": 1.1,
            "no_claim_when_uncertainty_overlaps_threshold": True,
            "warm_training_step_median_ratio_pr_over_main_max": 1.1,
            "marginal_joint_fit_over_added_independent_fit_max": 0.5,
            "startup_material_regression_report_ratio": 2.0,
            "warm_predict_material_regression_report_ratio": 1.5,
            "process_peak_rss_material_regression_report_ratio": 1.5,
        },
        "gated_representative_cases": ["resnet-feature-heavy", "unet-feature-heavy"],
        "joint_species_counts": [1, 2, 3, 4],
    }
    settings["cases"] = [
        {
            "name": "resnet-direct-one",
            "family": "resnet",
            "objective": "direct_log1p",
            "regions": 1,
            "side": 16,
            "channels": 3,
            "patch_size": 4,
            "units": 8,
        },
        {
            "name": "resnet-direct-multi",
            "family": "resnet",
            "objective": "direct_log1p",
            "regions": 2,
            "side": 16,
            "channels": 3,
            "patch_size": 4,
            "units": 8,
        },
        {
            "name": "resnet-mlpe-one",
            "family": "resnet",
            "objective": "mlpe",
            "regions": 1,
            "side": 16,
            "channels": 3,
            "patch_size": 4,
            "units": 8,
        },
        {
            "name": "resnet-mlpe-multi",
            "family": "resnet",
            "objective": "mlpe",
            "regions": 2,
            "side": 16,
            "channels": 3,
            "patch_size": 4,
            "units": 8,
        },
        {
            "name": "unet-direct-one",
            "family": "unet",
            "objective": "direct_log1p",
            "regions": 1,
            "side": 16,
            "channels": 3,
            "patch_size": 4,
            "units": 8,
        },
        {
            "name": "unet-direct-multi",
            "family": "unet",
            "objective": "direct_log1p",
            "regions": 2,
            "side": 16,
            "channels": 3,
            "patch_size": 4,
            "units": 8,
        },
        {
            "name": "unet-mlpe-one",
            "family": "unet",
            "objective": "mlpe",
            "regions": 1,
            "side": 16,
            "channels": 3,
            "patch_size": 4,
            "units": 8,
        },
        {
            "name": "unet-mlpe-multi",
            "family": "unet",
            "objective": "mlpe",
            "regions": 2,
            "side": 16,
            "channels": 3,
            "patch_size": 4,
            "units": 8,
        },
        {
            "name": "resnet-feature-heavy",
            "family": "resnet",
            "objective": "direct_log1p",
            "regions": 1,
            "side": 64,
            "channels": 16,
            "patch_size": 8,
            "units": 12,
        },
        {
            "name": "unet-feature-heavy",
            "family": "unet",
            "objective": "direct_log1p",
            "regions": 1,
            "side": 64,
            "channels": 16,
            "patch_size": 8,
            "units": 12,
        },
        {
            "name": "resnet-solver-stress",
            "family": "resnet",
            "objective": "mlpe",
            "regions": 1,
            "side": 64,
            "channels": 4,
            "patch_size": 4,
            "units": 48,
        },
        {
            "name": "unet-mlpe-stress",
            "family": "unet",
            "objective": "mlpe",
            "regions": 1,
            "side": 64,
            "channels": 4,
            "patch_size": 4,
            "units": 128,
        },
    ]
    return settings


def protocol_digest():
    """Identify workload settings in generated measurements."""
    return hashlib.sha256(json.dumps(protocol(), sort_keys=True).encode()).hexdigest()


def fixture(case, settings):
    """Return NumPy arrays and labels only; no library compatibility layer."""
    output = []
    grid_side = case["side"] // case["patch_size"]
    nodes = np.linspace(0, grid_side**2 - 1, case["units"], dtype=np.int64)
    positions = np.column_stack(np.unravel_index(nodes, (grid_side, grid_side)))
    positions = positions * case["patch_size"]
    left, right = np.triu_indices(case["units"], 1)
    for index in range(case["regions"]):
        rng = np.random.default_rng(settings["data_seed"] + index)
        features = rng.normal(size=(case["side"], case["side"], case["channels"])).astype(
            np.float32
        )
        separation = np.linalg.norm(positions[left] - positions[right], axis=1)
        targets = 0.2 + separation / case["side"] + rng.uniform(0.0, 0.03, size=len(left))
        ids = tuple(f"region-{index}-unit-{unit:03d}" for unit in range(case["units"]))
        pairs = tuple((ids[a], ids[b]) for a, b in zip(left, right, strict=True))
        digest = hashlib.sha256()
        digest.update(json.dumps({"ids": ids, "pairs": pairs}, sort_keys=True).encode())
        for array in (features, positions, targets):
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        output.append(
            dict(
                name=f"region-{index}",
                features=features,
                ids=ids,
                positions=positions,
                pairs=pairs,
                targets=targets,
                sha256=digest.hexdigest(),
            )
        )
    return output


def summary(samples):
    """Raw observations plus descriptive uncertainty, never a CI assertion."""
    values = np.asarray(samples, dtype=np.float64)
    rng = np.random.default_rng(52053)
    medians = np.median(rng.choice(values, size=(10000, len(values)), replace=True), axis=1)
    return dict(
        samples_seconds=values.tolist(),
        median_seconds=float(np.median(values)),
        minimum_seconds=float(values.min()),
        maximum_seconds=float(values.max()),
        sample_std_seconds=float(np.std(values, ddof=1)) if len(values) > 1 else None,
        median_bootstrap_95_percent_seconds=np.quantile(medians, [0.025, 0.975]).tolist(),
        bootstrap_resamples=10000,
        bootstrap_seed=52053,
    )
