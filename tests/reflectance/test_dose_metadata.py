#!/usr/bin/env python3
"""Unit tests for reflectance.dose_metadata module."""

import pytest
from src.reflectance.dose_metadata import (
    DoseRecord,
    iter_dose_records,
    dose_mapping,
    attach_dose_metadata,
)


@pytest.mark.quick
class TestDoseRecord:
    """Tests for DoseRecord dataclass."""

    def test_dose_record_creation(self):
        """Test DoseRecord can be created with valid data."""
        record = DoseRecord("dose_1", 0.0, 0.0)
        assert record.label == "dose_1"
        assert record.uva_mw_cm2 == 0.0
        assert record.uvb_mw_cm2 == 0.0

    def test_dose_record_as_tuple(self):
        """Test as_tuple property returns correct tuple."""
        record = DoseRecord("dose_2", 0.647, 0.246)
        assert record.as_tuple == (0.647, 0.246)

    def test_dose_record_frozen(self):
        """Test DoseRecord is immutable (frozen)."""
        record = DoseRecord("dose_1", 0.0, 0.0)
        with pytest.raises(AttributeError):
            record.label = "dose_2"


@pytest.mark.quick
class TestDoseFunctions:
    """Tests for dose metadata functions."""

    def test_iter_dose_records(self):
        """Test iter_dose_records returns 6 doses in order."""
        records = list(iter_dose_records())
        assert len(records) == 6
        assert records[0].label == "dose_1"
        assert records[5].label == "dose_6"

    def test_dose_records_ascending_uva(self):
        """Test dose records are in ascending UVA order."""
        records = list(iter_dose_records())
        uva_values = [r.uva_mw_cm2 for r in records]
        assert uva_values == sorted(uva_values)

    def test_dose_mapping(self):
        """Test dose_mapping returns dict with all 6 doses."""
        mapping = dose_mapping()
        assert len(mapping) == 6
        assert "dose_1" in mapping
        assert "dose_6" in mapping
        assert isinstance(mapping["dose_1"], DoseRecord)

    def test_dose_mapping_keys(self):
        """Test dose_mapping has correct keys."""
        mapping = dose_mapping()
        expected_keys = [f"dose_{i}" for i in range(1, 7)]
        assert set(mapping.keys()) == set(expected_keys)

    def test_attach_dose_metadata_valid(self):
        """Test attach_dose_metadata returns correct record for valid label."""
        record = attach_dose_metadata("dose_1")
        assert record.label == "dose_1"
        assert record.uva_mw_cm2 == 0.0
        assert record.uvb_mw_cm2 == 0.0

    def test_attach_dose_metadata_invalid(self):
        """Test attach_dose_metadata raises KeyError for invalid label."""
        with pytest.raises(KeyError, match="Unknown dose label: 'dose_99'"):
            attach_dose_metadata("dose_99")

    def test_dose_1_is_control(self):
        """Test dose_1 is the control (0, 0)."""
        record = attach_dose_metadata("dose_1")
        assert record.uva_mw_cm2 == 0.0
        assert record.uvb_mw_cm2 == 0.0

    def test_dose_values_match_known(self):
        """Test specific dose values match known calibration."""
        dose_2 = attach_dose_metadata("dose_2")
        assert dose_2.uva_mw_cm2 == 0.647
        assert dose_2.uvb_mw_cm2 == 0.246

        dose_5 = attach_dose_metadata("dose_5")
        assert dose_5.uva_mw_cm2 == 2.488
        assert dose_5.uvb_mw_cm2 == 0.768
