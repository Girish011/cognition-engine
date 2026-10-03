"""Unit tests for steering-vector math (numpy only, no model required)."""

import numpy as np
import pytest

from cognition_engine.steering.compute_vectors import (
    compute_steering_vectors,
    find_best_layer,
    load_steering_vectors,
    save_steering_vectors,
)
from cognition_engine.steering.contrastive_pairs import get_all_pairs


def test_vector_is_positive_minus_negative_mean():
    pos = np.ones((4, 3, 5))
    neg = np.zeros((4, 3, 5))
    vec = compute_steering_vectors({"positive": pos, "negative": neg})
    assert vec.shape == (3, 5)
    assert np.allclose(vec, 1.0)


def test_shape_mismatch_and_missing_keys_raise():
    with pytest.raises(ValueError):
        compute_steering_vectors(
            {"positive": np.zeros((2, 3, 4)), "negative": np.zeros((2, 3, 5))}
        )
    with pytest.raises(KeyError):
        compute_steering_vectors({"positive": np.zeros((2, 3, 4))})


def test_find_best_layer_picks_largest_norm():
    vecs = np.zeros((4, 8))
    vecs[2] = 3.0
    assert find_best_layer(vecs) == 2


def test_save_load_roundtrip(tmp_path):
    vecs = np.random.default_rng(0).normal(size=(3, 6))
    save_steering_vectors(vecs, "demo", output_dir=str(tmp_path))
    assert np.allclose(load_steering_vectors("demo", str(tmp_path)), vecs)


def test_contrastive_pair_sets_are_well_formed():
    pairs = get_all_pairs()
    assert set(pairs) == {"task_focus", "tool_format", "code_coherence"}
    for name, items in pairs.items():
        assert len(items) >= 10, name
        for p in items:
            assert p["positive"] != p["negative"]
