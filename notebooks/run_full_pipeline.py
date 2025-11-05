"""Regenerate processed tables and artifacts from raw datasets."""

from __future__ import annotations

import sys
from pathlib import Path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    src_root = project_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from pipelines import run_full_pipeline  # noqa: E402

    run_full_pipeline(project_root)
