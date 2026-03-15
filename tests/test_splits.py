"""Tests for benchmark split strategies."""

import pytest

from tsnn.data.splits.random_split import random_split
from tsnn.data.splits.cold_protein import cold_protein_split
from tsnn.data.splits.cold_scaffold import cold_scaffold_split


def test_random_split_sizes():
    ids = [f"complex_{i}" for i in range(100)]
    train, val, test = random_split(ids, train_ratio=0.7, val_ratio=0.15)

    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15
    assert len(set(train) | set(val) | set(test)) == 100  # No overlap


def test_random_split_deterministic():
    ids = [f"complex_{i}" for i in range(50)]
    train1, val1, test1 = random_split(ids, seed=42)
    train2, val2, test2 = random_split(ids, seed=42)

    assert train1 == train2
    assert val1 == val2
    assert test1 == test2


def test_cold_protein_no_overlap():
    ids = [f"complex_{i}" for i in range(50)]
    sequences = {f"complex_{i}": f"ACDEFGHIKLMNPQRSTVWY{'A' * i}" for i in range(50)}

    train, val, test = cold_protein_split(
        ids, sequences, identity_threshold=0.3
    )

    assert len(set(train) & set(test)) == 0, "No overlap between train and test"
    assert len(set(train) & set(val)) == 0, "No overlap between train and val"
    total = len(train) + len(val) + len(test)
    assert total == 50


def test_cold_scaffold_split():
    ids = [f"complex_{i}" for i in range(30)]
    smiles = {f"complex_{i}": f"C{'C' * (i % 5)}O" for i in range(30)}

    train, val, test = cold_scaffold_split(ids, smiles)

    total = len(train) + len(val) + len(test)
    assert total == 30
    assert len(set(train) & set(test)) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
