"""First classical HDR comparison on an original Kalantari scene.

This supplied starting implementation has two controls, not a new deghosting
method or a full Debevec reproduction. It uses no alignment or motion mask.
Review its formulas before using its outputs in a report. No training occurs.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity


def read_rgb(path):
    x = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if x is None or x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"Expected a 3-channel image: {path}")
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    if not np.isfinite(x).all():
        raise ValueError(f"Nonfinite data: {path}")
    return x


def mu_map(x):
    return np.log1p(5000.0 * x.astype(np.float64)) / np.log1p(5000.0)


def reference_only(radiance):
    return radiance[1].copy()


def weighted_merge(ldr, radiance):
    # A triangular weight favors encoded values near the middle of [0,1].
    # Weights apply separately to each color channel, without motion rejection.
    weights = np.maximum(0.0, 1.0 - 2.0 * np.abs(ldr - 0.5))
    denominator = weights.sum(axis=0)
    numerator = (weights * radiance).sum(axis=0)
    result = reference_only(radiance)
    np.divide(numerator, denominator, out=result, where=denominator > 1e-8)
    return result


def evaluate(prediction, target):
    if (not np.isfinite(prediction).all() or np.any(prediction < 0)
            or np.any(prediction > 1.0 + 1e-6)):
        raise ValueError("Prediction outside the declared finite [0,1] HDR convention; investigate scale.")
    a, b = mu_map(prediction), mu_map(target)
    mse = float(np.mean((a - b) ** 2))
    psnr = float(-10.0 * np.log10(mse)) if mse > 0 else float("inf")
    ssim = float(structural_similarity(a, b, channel_axis=-1, data_range=1.0,
                                      win_size=7, gaussian_weights=False,
                                      use_sample_covariance=True))
    return psnr, ssim


def display_map(x, reference_log_average):
    # One shared display exposure, computed from reference radiance ONLY.
    # The same mapping is applied to both predictions and the target.
    rgb = x.astype(np.float64)
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722])
    scaled_rgb = 0.18 * rgb / reference_log_average
    scaled_luminance = 0.18 * luminance / reference_log_average
    return np.clip(scaled_rgb / (1.0 + scaled_luminance[..., None]), 0, 1) ** (1.0 / 2.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"),
                        help="Optional identical zoom crop for all methods, in original pixel coordinates.")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output folder exists. Choose a new name to preserve the earlier result.")
    if not args.scene.is_dir():
        parser.error(f"Scene folder missing: {args.scene}")
    try:
        files = sorted(p for p in args.scene.iterdir() if p.suffix.lower() in (".tif", ".tiff"))
        exponents = np.loadtxt(args.scene / "exposure.txt", dtype=np.float64).reshape(-1)
        if len(files) != 3 or exponents.size != 3 or not np.isfinite(exponents).all():
            raise ValueError("Expected three TIFFs and three finite exposure exponents.")
        times = np.exp2(exponents)
        if not np.isfinite(times).all() or np.any(times <= 0):
            raise ValueError("Invalid relative exposure times.")
        records = sorted(zip(files, times), key=lambda item: item[1])
        images = []
        for path, _ in records:
            raw = read_rgb(path)
            if not np.issubdtype(raw.dtype, np.unsignedinteger):
                raise ValueError(f"Expected unsigned integer TIFF: {path}")
            images.append(raw.astype(np.float32) / np.iinfo(raw.dtype).max)
        ldr = np.stack(images)
        times = np.array([row[1] for row in records], dtype=np.float32)
        radiance = ldr ** 2.2 / times[:, None, None, None]
        target = read_rgb(args.scene / "HDRImg.hdr").astype(np.float32)
        if target.shape != ldr.shape[1:] or not np.all((target >= 0) & (target <= 1)):
            raise ValueError("Target must match dimensions and the declared [0,1] radiance convention.")
        h, w = target.shape[:2]
        if min(h, w) < 7:
            raise ValueError("SSIM requires dimensions >= 7.")
        crop = None
        if args.crop:
            x, y, cw, ch = args.crop
            if x < 0 or y < 0 or cw <= 0 or ch <= 0 or x + cw > w or y + ch > h:
                raise ValueError("Crop lies outside image bounds.")
            crop = (x, y, cw, ch)
        # Check the reference/target scale, but DO NOT fit or apply a scale factor.
        reference = reference_only(radiance)
        probe = ((ldr[1] > 0.1) & (ldr[1] < 0.9) & (reference > 1e-8) & (target > 0))
        ratios = target[probe] / reference[probe]
        scale_probe = np.percentile(ratios, [10, 50, 90]).tolist() if ratios.size else None
        predictions = {}
        rows = []
        for name, fn in [("reference_only", lambda: reference_only(radiance)),
                         ("weighted_merge_no_alignment", lambda: weighted_merge(ldr, radiance))]:
            start = time.perf_counter()
            prediction = fn()
            elapsed = time.perf_counter() - start
            psnr, ssim = evaluate(prediction, target)
            predictions[name] = prediction
            rows.append({"scene_id": args.scene.name, "method": name, "psnr_mu": psnr,
                         "ssim_mu": ssim, "merge_seconds_excluding_io_and_metrics": elapsed})
        args.out.mkdir(parents=True, exist_ok=False)
        for name, prediction in predictions.items():
            np.save(args.out / f"{name}.npy", prediction)
        with (args.out / "metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        config = {"status": "completed", "scene_id": args.scene.name,
                  "platform": platform.platform(), "python": sys.version.split()[0],
                  "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                  "numpy_version": np.__version__, "opencv_version": cv2.__version__,
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "evaluation_status": "Preliminary training-scene diagnostic; not held-out or final test performance",
                  "methods": [row["method"] for row in rows],
                  "source_files": [p.name for p, _ in records], "relative_times": times.tolist(),
                  "gamma": 2.2, "radiance_conversion": "I**2.2 / (2**e); unsigned integer dtype normalization",
                  "weighting": "Per-channel triangular weights: max(0, 1 - 2*abs(I-0.5)); reference fallback if denominator <=1e-8",
                  "metric_domain": "mu-law (mu=5000) on raw [0,1] radiance, no per-output normalization",
                  "psnr": "10*log10(1/MSE) across all pixels/channels", "ssim": "RGB mean; uniform 7x7 window; sample covariance; data_range=1",
                  "roi": "Whole image, no border excluded", "crop_display_only_xywh": crop,
                  "reference_target_ratio_probe_p10_p50_p90": scale_probe,
                  "probe_note": "Unsaturated reference channels; diagnostic only, may include scene/response mismatches; no fitted scaling applied",
                  "code_attribution": "Provided starting baseline implementation; review and cite appropriately",
                  "display": "Shared global Reinhard mapping with key 0.18 and reference-only geometric mean, followed by gamma 1/2.2",
                  "timing_note": "One CPU timing per method; not a hardware benchmark"}
        (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        ref_luma = reference.astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722])
        log_average = max(float(np.exp(np.mean(np.log(ref_luma + 1e-6)))), 1e-6)
        panels = [("Reference only", reference), ("Weighted merge / no alignment", predictions["weighted_merge_no_alignment"]),
                  ("Dataset HDR target", target)]
        nrows = 2 if crop else 1
        fig, axes = plt.subplots(nrows, 3, figsize=(16, 6 if nrows == 1 else 10), squeeze=False)
        for col, (title, array) in enumerate(panels):
            view = display_map(array, log_average)
            scale = min(1.0, 900 / max(h, w))
            small_view = cv2.resize(view, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            # Resampling may exceed display bounds by roundoff; clip previews only.
            axes[0, col].imshow(np.clip(small_view, 0, 1))
            axes[0, col].set_title(title, fontsize=12)
            axes[0, col].axis("off")
            if crop:
                x, y, cw, ch = crop
                axes[0, col].add_patch(plt.Rectangle((x * scale, y * scale), cw * scale, ch * scale,
                                                   fill=False, edgecolor="red", linewidth=1.5))
                axes[1, col].imshow(view[y:y+ch, x:x+cw])
                axes[1, col].set_title(f"Same crop x={x}, y={y}, w={cw}, h={ch}", fontsize=10)
                axes[1, col].axis("off")
        fig.suptitle(f"Scene {args.scene.name} / first classical controls / preliminary training-scene diagnostic", fontsize=14)
        fig.text(0.5, 0.03, "One shared display mapping from reference radiance. Scores use mu-law raw radiance; no target-fitted scaling.", ha="center", fontsize=10)
        fig.tight_layout(rect=(0, 0.07, 1, 0.93))
        fig.savefig(args.out / "comparison.png", dpi=150)
        plt.close(fig)
        for row in rows:
            print(f"{row['method']}: PSNR-mu={row['psnr_mu']:.4f} dB, SSIM-mu={row['ssim_mu']:.6f}")
        print("Reference-to-target scale probe (target/reference), p10/median/p90:", scale_probe)
        print("Probe only: these values did not rescale predictions.")
        print(f"Saved {args.out / 'comparison.png'}, metrics.csv, config.json, and raw prediction .npy files.")
        print("This is one training-scene diagnostic, not held-out model performance.")
    except (OSError, ValueError, cv2.error) as exc:
        parser.exit(1, f"Baseline run failed: {exc}\n")


if __name__ == "__main__":
    main()
