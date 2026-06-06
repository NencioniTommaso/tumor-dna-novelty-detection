"""
reference_io.py
Persistence helpers for saving and loading pre-computed reference distribution artifacts.
Analogous to model_io.py but for OA reference distributions (no SVM).
"""

import logging
import os

import joblib

logger = logging.getLogger(__name__)


def save_reference(state: dict, save_path: str) -> None:
    """Persist a reference distribution artifact to disk.

    Parameters
    ----------
    state : dict
        Must contain at minimum: 'xs', 'y_intra', 'K_train', 'train_states',
        'ref_seed', 'max_k', 'mismatches', 'mkl_weights'.
    save_path : str
        Absolute path where the .pkl file will be written.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    logger.info(f"Saving reference artifact to {save_path}...")
    joblib.dump(state, save_path)
    logger.info("Reference artifact saved successfully!")


def load_reference(ref_path: str) -> dict:
    """Load a previously saved reference distribution artifact.

    Returns
    -------
    dict
        The full state dictionary that was saved by save_reference().
    """
    if not os.path.exists(ref_path):
        logger.error(f"Reference file not found at {ref_path}")
        raise FileNotFoundError(ref_path)

    logger.info(f"Loading reference artifact from {ref_path}...")
    state = joblib.load(ref_path)
    logger.info(
        f"Reference loaded (seed={state.get('ref_seed')}, "
        f"max_k={state.get('max_k')}, m={state.get('mismatches')}, "
        f"train_files={[os.path.basename(f) for f in state.get('train_files', [])]})"
    )
    return state
