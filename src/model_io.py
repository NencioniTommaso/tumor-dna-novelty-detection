"""
model_io.py
Persistence helpers for saving and loading precomputed-kernel OC-SVM artifacts.
Uses a ModelArtifact dataclass to formalize the serialization schema.
"""

import dataclasses
import logging
import os
from typing import Optional

import joblib
from sklearn.svm import OneClassSVM

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ModelArtifact:
    """Typed container for a precomputed-kernel OC-SVM model artifact.

    This dataclass makes the save/load contract explicit, provides IDE
    autocomplete, and prevents key-typo bugs that arise with raw dicts.
    """
    model: OneClassSVM
    train_sequences: list[str]
    max_k: int
    mismatches: int
    nu_param: float
    mkl_weights: Optional[list[float]] = None
    train_states: Optional[dict] = None


def save_svm_model(artifact: ModelArtifact, save_path: str) -> None:
    """Serialize a ModelArtifact to disk using joblib."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    logger.info(f"Saving SVM state to {save_path}...")
    joblib.dump(dataclasses.asdict(artifact), save_path)
    logger.info("Model saved successfully!")


def load_svm_model(model_path: str) -> ModelArtifact:
    """
    Load a precomputed-kernel OC-SVM artifact.

    Returns a ModelArtifact with all fields populated.
    """
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(model_path)

    logger.info(f"Loading SVM state from {model_path}...")
    saved_state = joblib.load(model_path)

    return ModelArtifact(
        model=saved_state.get('model'),
        train_sequences=saved_state.get('train_sequences', []),
        max_k=saved_state.get('max_k'),
        mismatches=saved_state.get('mismatches'),
        nu_param=saved_state.get('nu_param'),
        mkl_weights=saved_state.get('mkl_weights'),
        train_states=saved_state.get('train_states'),
    )