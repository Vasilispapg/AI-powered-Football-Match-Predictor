"""Tests for model provenance stamping."""

from __future__ import annotations

import hashlib

import numpy as np

from football import provenance


class TestChecksum:
    def test_matches_hashlib(self, tmp_path) -> None:
        path = tmp_path / "data.csv"
        path.write_bytes(b"home,away\nA,B\n")

        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        assert provenance.file_checksum(path) == expected

    def test_changes_when_content_changes(self, tmp_path) -> None:
        path = tmp_path / "data.csv"
        path.write_bytes(b"one")
        first = provenance.file_checksum(path)
        path.write_bytes(b"two")

        assert provenance.file_checksum(path) != first

    def test_missing_file_is_none_not_an_error(self, tmp_path) -> None:
        assert provenance.file_checksum(tmp_path / "absent.csv") is None


class TestGitRevision:
    def test_returns_the_expected_shape(self) -> None:
        """Must not raise when git is missing -- that is normal, not an error."""
        revision = provenance.git_revision()

        assert set(revision) == {"commit", "dirty", "branch"}
        assert revision["commit"] is None or isinstance(revision["commit"], str)
        assert revision["dirty"] is None or isinstance(revision["dirty"], bool)


class TestCollect:
    def test_includes_everything_worth_recording(self, tmp_path) -> None:
        dataset = tmp_path / "matches.csv"
        dataset.write_text("a,b\n1,2\n")

        collected = provenance.collect(dataset)

        assert collected["seed"] == provenance.SEED
        assert collected["dataset_sha256"] is not None
        assert "git" in collected


class TestSeeds:
    def test_seeding_makes_numpy_reproducible(self) -> None:
        provenance.set_seeds(123)
        first = np.random.random(5)
        provenance.set_seeds(123)

        assert np.allclose(first, np.random.random(5))

    def test_different_seeds_differ(self) -> None:
        provenance.set_seeds(1)
        first = np.random.random(5)
        provenance.set_seeds(2)

        assert not np.allclose(first, np.random.random(5))
