#!/usr/bin/env python3
"""Integration tests for verifying module imports and basic functionality."""

import pytest


@pytest.mark.quick
class TestModuleImports:
    """Verify that all core modules can be imported successfully."""

    def test_import_chromatography(self):
        """Test that chromatography module imports successfully."""
        from src.chromatography import control_normalization_utils
        assert hasattr(control_normalization_utils, 'FORMS')
        assert hasattr(control_normalization_utils, '_summary_stats')

    def test_import_reflectance(self):
        """Test that reflectance module imports successfully."""
        from src.reflectance import dose_metadata
        assert hasattr(dose_metadata, 'DoseRecord')
        assert hasattr(dose_metadata, 'dose_mapping')

    def test_dose_metadata_accessible(self):
        """Test that dose metadata can be accessed and used."""
        from src.reflectance.dose_metadata import attach_dose_metadata
        
        dose_1 = attach_dose_metadata("dose_1")
        assert dose_1.label == "dose_1"
        assert dose_1.uva_mw_cm2 == 0.0
        assert dose_1.uvb_mw_cm2 == 0.0

    def test_forms_constant(self):
        """Test that FORMS constant is accessible and correct."""
        from src.chromatography.control_normalization_utils import FORMS
        
        assert isinstance(FORMS, list)
        assert "total" in FORMS
        assert "oxidized" in FORMS
        assert "reduced" in FORMS
