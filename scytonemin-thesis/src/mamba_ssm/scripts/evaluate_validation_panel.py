#!/usr/bin/env python3
"""
Audit the fixed validation panel against the Act_of_God readiness gates.

For each candidate checkpoint (EMA weights preferred) this script:
- Verifies the validation panel manifest hash.
- Re-runs deterministic inference (no eval-time augmentation) and computes per-spectrum PSNR/SAM after inverting any preprocessing.
- Checks the aggregate, tail, ROI, and optional downstream proxy gates.
- Logs per-group metrics for the three hold-out cohorts.
- Ranks checkpoints by the panel composite score (PSNR_mean − PSNR_std) and reports the winner.

Outputs:
- JSON summary with pass/fail for each gate, manifest hash, git hash, and composite score.
- CSV of per-spectrum metrics (PSNR, SAM, ROI stats, group metadata).
- CSV of per-group summaries.
- Optional downstream proxy results (if configured).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets import SpectraDataset, SpectraDatasetConfig  # noqa: E402
from src.models.mamba_uv import build_model  # noqa: E402
from src.roi_metrics import compute_dip_metrics  # noqa: E402


DEFAULT_MANIFEST = "data/validation_panel.csv"
DEFAULT_ROOT_DIR = "data/spectra_for_fold"
DEFAULT_OUTPUT_DIR = "final_analytics/panel_eval"
DEFAULT_CHECKPOINT_DIR = "checkpoints/god_run"
DEFAULT_WAVELENGTH_GRID = "data/spectra_for_fold/wavelength_grid.npy"
EPS = 1e-12


@dataclass
class PanelMetrics:
    psnr_mean: float
    psnr_std: float
    sam_mean: float
    pass_gate: bool


@dataclass
class TailMetrics:
    psnr_pass_ratio: float
    sam_pass_ratio: float
    joint_pass_ratio: float
    pass_gate: bool


@dataclass
class ROIMetrics:
    area_within_10_pct_ratio: float
    centroid_mae_nm_median: float
    depth_mae_median: float
    spurious_peaks: int
    pass_gate: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help="Validation panel manifest CSV (default: data/validation_panel.csv)",
    )
    parser.add_argument(
        "--expected-hash",
        default=None,
        help="Expected SHA-256 hash of the panel manifest (hex). Fails if mismatch.",
    )
    parser.add_argument(
        "--root-dir",
        default=DEFAULT_ROOT_DIR,
        help="Root directory for spectra (.npy files) (default: data/spectra_for_fold)",
    )
    parser.add_argument(
        "--baseline-dir",
        default=None,
        help="Directory with baseline/raw spectra; defaults to --root-dir.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional metadata CSV for downstream proxy/dose checks.",
    )
    parser.add_argument(
        "--wavelength-grid",
        default=DEFAULT_WAVELENGTH_GRID,
        help="Wavelength grid .npy for downstream proxy (default: data/spectra_for_fold/wavelength_grid.npy).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Explicit checkpoint to evaluate. If omitted, discover top-K in --checkpoint-dir.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing candidate checkpoints (default: checkpoints/god_run)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Evaluate the K most recent checkpoints (default: 3). Ignored if --checkpoint supplied.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store evaluation outputs (default: final_analytics/panel_eval)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override (cpu/cuda:0, etc.). Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--downstream-script",
        default=None,
        help=(
            "Optional script to evaluate downstream proxy. Script must accept --pred-dir and --reference "
            "arguments and emit JSON with 'metrics' entries."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Inference batch size (default: 128).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (default: 0).",
    )
    return parser.parse_args()


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_hash(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            check=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def discover_checkpoints(args: argparse.Namespace) -> List[Path]:
    if args.checkpoint:
        ckpt = Path(args.checkpoint).resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint {ckpt} not found")
        return [ckpt]

    ckpt_dir = Path(args.checkpoint_dir).resolve()
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} not found")
    candidates = sorted(
        (p for p in ckpt_dir.glob("*.pt") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    if args.top_k <= 0:
        raise ValueError("--top-k must be >= 1")
    return candidates[: args.top_k]


def load_checkpoint(path: Path) -> Tuple[Dict, Dict]:
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint.get("config", {})
    if "model_state" not in checkpoint and "ema_state" not in checkpoint:
        raise KeyError(f"Checkpoint {path} missing model weights")
    return checkpoint, config


def resolve_path(base: Path, value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    candidate = Path(value)
    if candidate.exists():
        return candidate
    alt = (base / value).resolve()
    return alt if alt.exists() else None


def build_dataset(
    manifest: Path,
    root_dir: Path,
    config: Dict,
) -> SpectraDataset:
    film_features = config.get("film_features") or ["cos_theta"]
    dose_features_csv = resolve_path(REPO_ROOT, config.get("dose_features_csv"))
    dose_stats_json = resolve_path(REPO_ROOT, config.get("dose_stats_json"))

    cond_mean: Optional[Sequence[float]] = None
    cond_std: Optional[Sequence[float]] = None
    if dose_stats_json and dose_stats_json.exists():
        stats = json.loads(dose_stats_json.read_text())
        cond_mean = stats.get("mean")
        cond_std = stats.get("std")

    cfg = SpectraDatasetConfig(
        sequence_length=int(config.get("sequence_length", 601)),
        train_dir=str(root_dir),
        noise2noise=False,
        target_wavelengths=None,
        split="val",
        manifest_path=str(manifest),
        dose_features_path=str(dose_features_csv) if dose_features_csv else None,
        film_features=tuple(film_features),
        cond_mean=cond_mean,
        cond_std=cond_std,
    )
    return SpectraDataset(cfg)


def build_model_from_checkpoint(
    checkpoint: Dict,
    config: Dict,
    dataset: SpectraDataset,
    device: torch.device,
) -> torch.nn.Module:
    model_name = config.get("model", "mamba_tiny_uv")
    kwargs = {
        "sequence_length": dataset.lambda_norm.shape[0],
        "in_channels": 3,
        "out_activation": config.get("out_activation", "sigmoid"),
    }
    for key in ("d_model", "n_layers", "d_state", "d_conv", "expand"):
        value = config.get(key)
        if value is not None:
            kwargs[key] = value

    geometry_film = bool(config.get("geometry_film", True))
    if geometry_film:
        kwargs["geometry_film"] = True
        kwargs["film_hidden_dim"] = int(config.get("film_hidden_dim", 64))
        kwargs["cond_dim"] = dataset.conditioning[0].numel() if dataset.conditioning else 0

    model = build_model(model_name, **kwargs)
    state = checkpoint.get("ema_state") or checkpoint.get("model_state")
    state = {
        k.replace("module.", ""): v
        for k, v in state.items()
        if not k.endswith("n_averaged")
    }
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def forward_pass(
    model: torch.nn.Module,
    dataset: SpectraDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    cond_dim = dataset.conditioning[0].numel() if dataset.conditioning else 0

    with torch.no_grad():
        offset = 0
        for noisy, target, cond in loader:
            noisy = noisy.to(device)
            cond_tensor = cond.to(device) if cond_dim > 0 else None
            prediction = model(noisy, cond=cond_tensor) if cond_dim > 0 else model(noisy)
            prediction = prediction.squeeze(1).to(torch.float32).cpu().numpy()
            targ = target.squeeze(1).to(torch.float32).cpu().numpy()
            preds.append(prediction)
            targets.append(targ)
            offset += prediction.shape[0]

    pred_arr = np.concatenate(preds, axis=0)
    targ_arr = np.concatenate(targets, axis=0)
    return pred_arr, targ_arr


def psnr_per_sample(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> np.ndarray:
    mse = np.mean((pred - target) ** 2, axis=1)
    mse = np.clip(mse, EPS, None)
    max_val = float(data_range)
    return 20.0 * np.log10(max_val) - 10.0 * np.log10(mse)


def sam_per_sample(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    dot = np.sum(pred * target, axis=1)
    pred_norm = np.linalg.norm(pred, axis=1)
    targ_norm = np.linalg.norm(target, axis=1)
    cosine = np.clip(dot / (pred_norm * targ_norm + EPS), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def compute_aggregate_metrics(psnr: np.ndarray, sam: np.ndarray) -> PanelMetrics:
    psnr_mean = float(psnr.mean())
    psnr_std = float(psnr.std(ddof=0))
    sam_mean = float(sam.mean())
    pass_gate = (psnr_mean >= 18.0) and (psnr_std <= 4.0) and (sam_mean <= 9.0)
    return PanelMetrics(psnr_mean, psnr_std, sam_mean, pass_gate)


def compute_tail_metrics(psnr: np.ndarray, sam: np.ndarray) -> TailMetrics:
    psnr_pass = psnr >= 14.0
    sam_pass = sam <= 12.0
    joint_pass = psnr_pass & sam_pass
    return TailMetrics(
        psnr_pass_ratio=float(psnr_pass.mean()),
        sam_pass_ratio=float(sam_pass.mean()),
        joint_pass_ratio=float(joint_pass.mean()),
        pass_gate=bool(joint_pass.mean() >= 0.95),
    )


def compute_roi_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    lam_nm: np.ndarray,
    baseline: str = "local",
    baseline_guard_nm: float = 10.0,
    roi: Tuple[float, float] = (320.0, 500.0),
    max_area_error_pct: float = 10.0,
) -> ROIMetrics:
    area_errors: List[float] = []
    centroid_errors: List[float] = []
    depth_mae_values: List[float] = []
    spurious_peaks = 0

    for pred_i, target_i in zip(pred, target):
        summary, records = compute_dip_metrics(
            pred_i,
            target_i,
            lam_nm,
            roi=roi,
            baseline=baseline,
            baseline_guard_nm=baseline_guard_nm,
        )
        if records:
            area_med = np.median([rec["area_error_pct"] for rec in records])
            centroid_med = np.median([rec["centroid_error_nm"] for rec in records])
            depth_med = np.median([rec["depth_mae"] for rec in records])
        else:
            area_med = 0.0
            centroid_med = 0.0
            depth_med = 0.0
        area_errors.append(float(area_med))
        centroid_errors.append(float(centroid_med))
        depth_mae_values.append(float(depth_med))

        # Heuristic: spurious peaks flagged if we detected dips but target absorption was ~zero (area_med == 0).
        if records and float(np.median([rec["target_area"] for rec in records])) < 1e-6:
            spurious_peaks += 1

    area_array = np.asarray(area_errors, dtype=np.float32)
    centroid_array = np.asarray(centroid_errors, dtype=np.float32)
    depth_array = np.asarray(depth_mae_values, dtype=np.float32)

    within_area = float((area_array <= max_area_error_pct).mean())
    centroid_mae_nm_med = float(np.median(centroid_array))
    depth_mae_med = float(np.median(depth_array))
    pass_gate = (within_area >= 0.90) and (spurious_peaks == 0)

    return ROIMetrics(within_area, centroid_mae_nm_med, depth_mae_med, spurious_peaks, pass_gate)


def run_downstream_proxy(
    script_path: Path,
    manifest_path: Path,
    pred_dir: Path,
    baseline_dir: Path,
    metadata_path: Optional[Path],
    wavelength_grid: Optional[Path],
) -> Tuple[bool, Dict[str, object]]:
    output_json = pred_dir / "downstream_metrics.json"
    cmd = [
        sys.executable or 'python3',
        str(script_path),
        "--manifest",
        str(manifest_path),
        "--denoised-dir",
        str(pred_dir),
        "--baseline-dir",
        str(baseline_dir),
        "--output-json",
        str(output_json),
    ]
    if metadata_path:
        cmd.extend(["--metadata", str(metadata_path)])
    if wavelength_grid:
        cmd.extend(["--wavelength-grid", str(wavelength_grid)])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Downstream proxy script failed with code {result.returncode}: {result.stderr}"
        )
    if not output_json.exists():
        raise RuntimeError(
            f"Downstream proxy script did not produce output JSON at {output_json}"
        )
    payload = json.loads(output_json.read_text())
    pass_gate = bool(payload.get("pass", True))
    return pass_gate, payload


def save_per_spectrum_metrics(
    path: Path,
    metadata: List[Dict[str, str]],
    psnr: np.ndarray,
    sam: np.ndarray,
    area_errors: np.ndarray,
    centroid_errors: np.ndarray,
    depth_errors: np.ndarray,
) -> None:
    headers = [
        "relative_path",
        "group_id",
        "treatment",
        "sample",
        "angle",
        "psnr_db",
        "sam_deg",
        "roi_area_error_pct",
        "roi_centroid_error_nm",
        "roi_depth_mae",
    ]
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for meta, psnr_i, sam_i, area_i, centroid_i, depth_i in zip(
            metadata, psnr, sam, area_errors, centroid_errors, depth_errors
        ):
            writer.writerow(
                [
                    meta.get("relative_path", ""),
                    meta.get("group_id", ""),
                    meta.get("treatment", ""),
                    meta.get("sample", ""),
                    meta.get("angle", ""),
                    f"{psnr_i:.6f}",
                    f"{sam_i:.6f}",
                    f"{area_i:.6f}",
                    f"{centroid_i:.6f}",
                    f"{depth_i:.6f}",
                ]
            )


def per_group_summary(
    metadata: List[Dict[str, str]],
    psnr: np.ndarray,
    sam: np.ndarray,
    area_errors: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    by_group: Dict[str, List[int]] = {}
    for idx, meta in enumerate(metadata):
        group = meta.get("group_id", f"group_{idx}")
        by_group.setdefault(group, []).append(idx)

    summary: Dict[str, Dict[str, float]] = {}
    for group, indices in by_group.items():
        psnr_vals = psnr[indices]
        sam_vals = sam[indices]
        area_vals = area_errors[indices]
        summary[group] = {
            "count": float(len(indices)),
            "psnr_mean": float(psnr_vals.mean()),
            "psnr_std": float(psnr_vals.std(ddof=0)),
            "sam_mean": float(sam_vals.mean()),
            "sam_std": float(sam_vals.std(ddof=0)),
            "area_error_median": float(np.median(area_vals)),
            "area_error_90pct": float(np.percentile(area_vals, 90)),
        }
    return summary


def evaluate_checkpoint(
    ckpt_path: Path,
    manifest: Path,
    root_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict:
    checkpoint, config = load_checkpoint(ckpt_path)
    dataset = build_dataset(manifest, root_dir, config)
    model = build_model_from_checkpoint(checkpoint, config, dataset, device)

    pred, target = forward_pass(
        model,
        dataset,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if not (np.isfinite(pred).all() and np.isfinite(target).all()):
        raise ValueError("Non-finite values encountered in predictions or targets.")
    pred_min, pred_max = float(pred.min()), float(pred.max())
    targ_min, targ_max = float(target.min()), float(target.max())
    if (pred_min < -1e-3 or pred_max > 1 + 1e-3) or (targ_min < -1e-3 or targ_max > 1 + 1e-3):
        print(
            "[WARN] Predictions/targets extend outside [0, 1]; "
            "PSNR with data_range=1.0 may not reflect true quality."
        )

    # Inference already uses the same normalization; the predictions are in reflectance space [0, 1].
    psnr_values = psnr_per_sample(pred, target, data_range=1.0)
    sam_values = sam_per_sample(pred, target)

    lam_nm = dataset.lambda_grid.astype(np.float64)
    roi_metrics = compute_roi_metrics(
        pred,
        target,
        lam_nm,
        baseline=config.get("baseline", "local"),
        baseline_guard_nm=float(config.get("baseline_guard_nm", 10.0)),
    )

    aggregate = compute_aggregate_metrics(psnr_values, sam_values)
    tails = compute_tail_metrics(psnr_values, sam_values)

    # ROI stats need per-spectrum area errors etc. Recompute with storage for CSV export.
    area_errors: List[float] = []
    centroid_errors: List[float] = []
    depth_errors: List[float] = []
    for pred_i, target_i in zip(pred, target):
        summary, records = compute_dip_metrics(
            pred_i,
            target_i,
            lam_nm,
            roi=(320.0, 500.0),
            baseline=config.get("baseline", "local"),
            baseline_guard_nm=float(config.get("baseline_guard_nm", 10.0)),
        )
        if records:
            area_errors.append(float(np.median([rec["area_error_pct"] for rec in records])))
            centroid_errors.append(float(np.median([rec["centroid_error_nm"] for rec in records])))
            depth_errors.append(float(np.median([rec["depth_mae"] for rec in records])))
        else:
            area_errors.append(0.0)
            centroid_errors.append(0.0)
            depth_errors.append(0.0)

    metadata = dataset.metadata_rows
    group_summary = per_group_summary(metadata, psnr_values, sam_values, np.asarray(area_errors))

    composite = aggregate.psnr_mean - aggregate.psnr_std
    gate_pass = aggregate.pass_gate and tails.pass_gate and roi_metrics.pass_gate

    return {
        "checkpoint": str(ckpt_path),
        "aggregate": aggregate.__dict__,
        "tails": tails.__dict__,
        "roi": roi_metrics.__dict__,
        "per_group": group_summary,
        "psnr_values": psnr_values.tolist(),
        "sam_values": sam_values.tolist(),
        "area_errors": area_errors,
        "centroid_errors": centroid_errors,
        "depth_errors": depth_errors,
        "metadata": metadata,
        "predictions": pred,
        "targets": target,
        "composite_score": composite,
        "passed_all_gates": gate_pass,
    }


def main() -> None:
    args = parse_args()

    repo_root = REPO_ROOT
    manifest = (repo_root / args.manifest).resolve()
    root_dir = (repo_root / args.root_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not manifest.exists():
        raise FileNotFoundError(f"Manifest {manifest} not found")

    manifest_hash = compute_sha256(manifest)
    if args.expected_hash and manifest_hash.lower() != args.expected_hash.lower():
        raise ValueError(
            f"Manifest hash mismatch. Expected {args.expected_hash}, observed {manifest_hash}"
        )

    git_hash = get_git_hash(repo_root)

    checkpoints = discover_checkpoints(args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    eval_results: List[Dict] = []
    if args.baseline_dir:
        baseline_dir = resolve_path(repo_root, args.baseline_dir) or Path(args.baseline_dir).resolve()
    else:
        baseline_dir = root_dir
    if not baseline_dir.exists():
        raise FileNotFoundError(f"Baseline directory not found: {baseline_dir}")
    metadata_path = resolve_path(repo_root, args.metadata) if args.metadata else None
    wavelength_grid_path = resolve_path(repo_root, args.wavelength_grid) if args.wavelength_grid else None
    if wavelength_grid_path is None:
        raise FileNotFoundError(f"Wavelength grid not found: {args.wavelength_grid}")
    for ckpt_path in checkpoints:
        print(f"[INFO] Evaluating checkpoint {ckpt_path}")
        result = evaluate_checkpoint(
            ckpt_path=ckpt_path,
            manifest=manifest,
            root_dir=root_dir,
            device=device,
            args=args,
        )

        if args.downstream_script:
            pred_dir = output_dir / (Path(ckpt_path).stem + "_panel_preds")
            pred_dir.mkdir(parents=True, exist_ok=True)
            pred_array = np.asarray(result["predictions"], dtype=np.float32)
            for spec, meta in zip(pred_array, result["metadata"]):
                rel = Path(meta["relative_path"])
                out_path = pred_dir / rel.with_name(rel.stem + "_denoised.npy")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(out_path, spec)
            pass_gate, downstream_payload = run_downstream_proxy(
                Path(args.downstream_script),
                manifest,
                pred_dir,
                baseline_dir,
                metadata_path,
                wavelength_grid_path,
            )
            result["downstream"] = {
                "pass_gate": pass_gate,
                "metrics": downstream_payload.get("metrics", {}),
                "payload": downstream_payload,
                "output_dir": str(pred_dir),
            }
            if not pass_gate:
                result["passed_all_gates"] = False
        else:
            result["downstream"] = None

        eval_results.append(result)
        downstream_ok = result["downstream"]["pass_gate"] if result["downstream"] else True
        print(
            "[INFO] "
            f"{Path(ckpt_path).name}: PSNR_mean={result['aggregate']['psnr_mean']:.3f} "
            f"PSNR_std={result['aggregate']['psnr_std']:.3f} "
            f"SAM_mean={result['aggregate']['sam_mean']:.3f} "
            f"(aggregate={'PASS' if result['aggregate']['pass_gate'] else 'FAIL'}, "
            f"tails={'PASS' if result['tails']['pass_gate'] else 'FAIL'}, "
            f"roi={'PASS' if result['roi']['pass_gate'] else 'FAIL'}, "
            f"downstream={'PASS' if downstream_ok else 'FAIL'})"
        )

        # Persist per-spectrum metrics for this checkpoint.
        per_spec_path = output_dir / f"{Path(ckpt_path).stem}_per_spectrum_metrics.csv"
        save_per_spectrum_metrics(
            per_spec_path,
            result["metadata"],
            np.asarray(result["psnr_values"]),
            np.asarray(result["sam_values"]),
            np.asarray(result["area_errors"]),
            np.asarray(result["centroid_errors"]),
            np.asarray(result["depth_errors"]),
        )

        # Per-group summary CSV.
        per_group_path = output_dir / f"{Path(ckpt_path).stem}_per_group_summary.csv"
        with per_group_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["group_id", "count", "psnr_mean", "psnr_std", "sam_mean", "sam_std", "area_error_median", "area_error_90pct"]
            )
            for group_id, stats in result["per_group"].items():
                writer.writerow(
                    [
                        group_id,
                        f"{stats['count']:.0f}",
                        f"{stats['psnr_mean']:.4f}",
                        f"{stats['psnr_std']:.4f}",
                        f"{stats['sam_mean']:.4f}",
                        f"{stats['sam_std']:.4f}",
                        f"{stats['area_error_median']:.4f}",
                        f"{stats['area_error_90pct']:.4f}",
                    ]
                )

        # Clean up heavy numpy arrays before summary export.
        result.pop("predictions", None)
        result.pop("targets", None)

    passing = [res for res in eval_results if res["passed_all_gates"]]
    ranked_results = sorted(passing or eval_results, key=lambda r: r["composite_score"], reverse=True)
    best = ranked_results[0]

    per_checkpoint_summary = [
        {
            "checkpoint": res["checkpoint"],
            "composite_score": res["composite_score"],
            "aggregate_pass": res["aggregate"]["pass_gate"],
            "tail_pass": res["tails"]["pass_gate"],
            "roi_pass": res["roi"]["pass_gate"],
            "downstream_pass": (res["downstream"]["pass_gate"] if res["downstream"] else True),
            "passed_all_gates": res["passed_all_gates"],
            "psnr_mean": res["aggregate"]["psnr_mean"],
            "psnr_std": res["aggregate"]["psnr_std"],
            "sam_mean": res["aggregate"]["sam_mean"],
        }
        for res in ranked_results
    ]

    summary = {
        "manifest": str(manifest),
        "manifest_hash": manifest_hash,
        "git_hash": git_hash,
        "device": str(device),
        "args": vars(args),
        "versions": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "checkpoints_evaluated": [res["checkpoint"] for res in eval_results],
        "passing_checkpoints": [res["checkpoint"] for res in passing],
        "ranked_order": [res["checkpoint"] for res in ranked_results],
        "best_checkpoint": best["checkpoint"],
        "best_composite_score": best["composite_score"],
        "best_passed_all_gates": best["passed_all_gates"],
        "per_checkpoint": per_checkpoint_summary,
        "gates": {
            "aggregate_pass": best["aggregate"]["pass_gate"],
            "tail_pass": best["tails"]["pass_gate"],
            "roi_pass": best["roi"]["pass_gate"],
            "downstream_pass": (best["downstream"]["pass_gate"] if best["downstream"] else True),
        },
        "aggregate_metrics": best["aggregate"],
        "tail_metrics": best["tails"],
        "roi_metrics": best["roi"],
        "per_group": best["per_group"],
    }

    summary_path = output_dir / "panel_eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print(f"[INFO] Summary written to {summary_path}")
    if best["passed_all_gates"]:
        print(f"[PASS] Readiness gates satisfied by {best['checkpoint']}")
    else:
        print(f"[FAIL] Readiness gates NOT satisfied by {best['checkpoint']}")


if __name__ == "__main__":
    print('[INFO] evaluate_validation_panel.py (Act_of_God QA) 2025-10-09 version active')
    main()
