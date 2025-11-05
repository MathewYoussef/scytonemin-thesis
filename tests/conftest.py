"""Test fixtures for the thesis repository."""
from pathlib import Path
import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]
