"""Pipeline entry points for sample/full data runs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


def _python_env(project_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    src_root = str(project_root / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([src_root, existing]) if existing else src_root
    return env


def _run_module(
    module: str,
    args: Sequence[str],
    *,
    project_root: Path,
    cwd: Path | None = None,
) -> None:
    cmd = [sys.executable, "-m", module, *args]
    subprocess.run(cmd, check=True, cwd=cwd or project_root, env=_python_env(project_root))


def _run_stage_calibrations(project_root: Path) -> None:
    config = project_root / "data" / "reference" / "initial_calibration" / "analysis_config.yaml"
    if not config.exists():
        raise FileNotFoundError(f"Initial calibration config not found: {config}")

    print("[pipelines] Stage A — calibrating chromatogram standards")
    _run_module(
        "chromatography.run_stage_a",
        ["--config", str(config)],
        project_root=project_root,
    )

    print("[pipelines] Stage B/C — applying calibrations to treatments")
    _run_module(
        "chromatography.run_stage_bc",
        ["--config", str(config)],
        project_root=project_root,
    )


def _run_reflectance_harmonisation(project_root: Path) -> None:
    output_dir = (project_root / "data" / "reference" / "reflectance" / "canonical_dataset").resolve()
    package_root = project_root / "src" / "reflectance"
    if not (package_root / "archive" / "aggregated_reflectance").exists():
        raise FileNotFoundError(
            "Expected reflectance archive at src/reflectance/archive/aggregated_reflectance."
        )

    print("[pipelines] Reflectance — rebuilding canonical dataset")
    _run_module(
        "reflectance.build_canonical_dataset",
        ["--output", str(output_dir)],
        project_root=project_root,
        cwd=package_root,
    )


def _run_mamba_validation(project_root: Path) -> None:
    try:
        import torch  # noqa: F401
        from mamba_ssm import Mamba  # type: ignore # noqa: F401
    except ImportError as exc:
        print(f"[pipelines] Skipping Mamba validation (missing dependency: {exc})")
        return

    manifest = project_root / "data" / "reference" / "mamba_ssm" / "validation_panel.csv"
    root_dir = project_root / "data" / "raw" / "mamba_ssm" / "spectra_for_fold"
    checkpoint_dir = project_root / "models" / "mamba_ssm" / "checkpoints" / "god_run"
    output_dir = project_root / "ops" / "output" / "data" / "mamba_checks" / "panel_eval"

    missing: list[tuple[str, Path]] = [
        ("manifest", manifest),
        ("spectra", root_dir),
        ("checkpoints", checkpoint_dir),
    ]
    for label, path in missing:
        if not path.exists():
            raise FileNotFoundError(f"Mamba {label} path not found: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("[pipelines] Mamba-SSM — validating Act-of-God checkpoint(s)")
    _run_module(
        "mamba_ssm.scripts.evaluate_validation_panel",
        [
            "--manifest",
            str(manifest),
            "--root-dir",
            str(root_dir),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--output-dir",
            str(output_dir.relative_to(project_root)),
        ],
        project_root=project_root,
    )


def run_sample_checks(project_root: Path) -> None:
    """Execute lightweight smoke checks on a constrained subset of the data."""
    config = project_root / "data" / "reference" / "initial_calibration" / "analysis_config.yaml"
    if not config.exists():
        raise FileNotFoundError(config)

    print("[pipelines] Sample check — Stage A (total form only)")
    _run_module(
        "chromatography.run_stage_a",
        ["--config", str(config), "--forms", "total"],
        project_root=project_root,
    )


def run_full_pipeline(project_root: Path) -> None:
    """Rebuild calibration tables, reflectance canonical datasets, and Mamba evaluations."""

    print(f"[pipelines] Starting full pipeline from {project_root}")
    _run_stage_calibrations(project_root)
    _run_reflectance_harmonisation(project_root)
    _run_mamba_validation(project_root)
    print("[pipelines] Full pipeline complete")


def render_all_figures(project_root: Path) -> None:
    """Regenerate figure bundles after data pipelines finish."""
    figures_root = project_root / "scaffold" / "mamba_ssm" / "figures" / "roi_local_plots_full"
    if not figures_root.exists():
        print("[pipelines] No figure regeneration implemented yet.")
        return
    print("[pipelines] Figure regeneration hooks pending implementation.")
