#!/usr/bin/env python3
"""Unit tests for chromatography.control_normalization_utils module."""

import pytest
import pandas as pd
import numpy as np
from src.chromatography.control_normalization_utils import (
    FORMS,
    _summary_stats,
)


@pytest.mark.quick
class TestConstants:
    """Tests for module constants."""

    def test_forms_list(self):
        """Test FORMS constant contains expected scytonemin forms."""
        assert FORMS == ["total", "oxidized", "reduced"]
        assert len(FORMS) == 3
        assert "total" in FORMS
        assert "oxidized" in FORMS
        assert "reduced" in FORMS


@pytest.mark.quick
class TestSummaryStats:
    """Tests for _summary_stats function."""

    def test_summary_stats_basic(self):
        """Test _summary_stats with basic numeric data."""
        values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = _summary_stats(values)
        
        assert stats["count"] == 5
        assert stats["median"] == 3.0
        assert stats["mean"] == 3.0
        assert stats["iqr"] == 2.0  # Q3(4) - Q1(2)

    def test_summary_stats_empty(self):
        """Test _summary_stats with empty series."""
        values = pd.Series([], dtype=float)
        stats = _summary_stats(values)
        
        assert stats["count"] == 0
        assert np.isnan(stats["median"])
        assert np.isnan(stats["mean"])
        assert np.isnan(stats["std"])
        assert np.isnan(stats["iqr"])

    def test_summary_stats_with_nan(self):
        """Test _summary_stats drops NaN values."""
        values = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
        stats = _summary_stats(values)
        
        assert stats["count"] == 3
        assert stats["median"] == 3.0
        assert stats["mean"] == 3.0

    def test_summary_stats_single_value(self):
        """Test _summary_stats with single value (std should be NaN)."""
        values = pd.Series([42.0])
        stats = _summary_stats(values)
        
        assert stats["count"] == 1
        assert stats["median"] == 42.0
        assert stats["mean"] == 42.0
        assert np.isnan(stats["std"])  # Can't compute std with n=1

    def test_summary_stats_two_values(self):
        """Test _summary_stats with two values (std should be computed)."""
        values = pd.Series([1.0, 3.0])
        stats = _summary_stats(values)
        
        assert stats["count"] == 2
        assert stats["median"] == 2.0
        assert stats["mean"] == 2.0
        assert not np.isnan(stats["std"])
        assert stats["std"] == pytest.approx(1.414, rel=0.01)

    def test_summary_stats_identical_values(self):
        """Test _summary_stats with identical values."""
        values = pd.Series([5.0, 5.0, 5.0, 5.0])
        stats = _summary_stats(values)
        
        assert stats["count"] == 4
        assert stats["median"] == 5.0
        assert stats["mean"] == 5.0
        assert stats["std"] == 0.0
        assert stats["iqr"] == 0.0

    def test_summary_stats_returns_dict(self):
        """Test _summary_stats returns dict with expected keys."""
        values = pd.Series([1.0, 2.0, 3.0])
        stats = _summary_stats(values)
        
        assert isinstance(stats, dict)
        expected_keys = {"median", "mean", "std", "iqr", "count"}
        assert set(stats.keys()) == expected_keys

    def test_summary_stats_types(self):
        """Test _summary_stats returns correct types."""
        values = pd.Series([1.0, 2.0, 3.0, 4.0])
        stats = _summary_stats(values)
        
        assert isinstance(stats["count"], int)
        assert isinstance(stats["median"], float)
        assert isinstance(stats["mean"], float)
        assert isinstance(stats["iqr"], float)
