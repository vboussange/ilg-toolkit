"""ILG Toolkit: inverse landscape genetics (ILG) in JAX."""

from .calibration import (
    AbstractCalibrationHead,
    DirectCalibrationHead,
    fit_calibration_head,
    MLPECalibrationHead,
)
from .config import ResistanceSolverConfig, TrainingConfig
from .data import PairwiseObservations, RegionBatch, TargetSpec
from .ensemble import Ensemble, EnsemblePrediction
from .laplace import (
    laplace_approximation,
    LaplaceApproximation,
    newton_step,
    summed_training_objective,
)
from .losses import AbstractLoss, Log1pLoss, MLPELoss
from .model import LandscapeGeneticsModel
from .penalties import RidgePenalty
from .persistence import ArtifactError, load_model, save_model
from .prediction import Prediction
from .training import StepRecord, train, TrainingState


__all__ = [
    "AbstractCalibrationHead",
    "AbstractLoss",
    "ArtifactError",
    "DirectCalibrationHead",
    "Ensemble",
    "EnsemblePrediction",
    "LandscapeGeneticsModel",
    "LaplaceApproximation",
    "Log1pLoss",
    "MLPECalibrationHead",
    "MLPELoss",
    "PairwiseObservations",
    "Prediction",
    "RegionBatch",
    "ResistanceSolverConfig",
    "RidgePenalty",
    "StepRecord",
    "TargetSpec",
    "TrainingConfig",
    "TrainingState",
    "fit_calibration_head",
    "laplace_approximation",
    "load_model",
    "newton_step",
    "save_model",
    "summed_training_objective",
    "train",
]
