"""Conservative reference-fallback HDR baseline; supplied starting implementation.

Requires classical_baselines.py in the same directory. Uses input images only
to build rejection masks. The dataset target is used exclusively for evaluation
and display. The masks indicate photometric disagreement or weak/clipped source
measurements, not ground-truth motion. No alignment or learning occurs.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

import classical_baselines as base


def reject_motion(ldr, radiance, threshold=0.2, low=0.05, high=0.95):
    """Fallback to reference if any source is suspect and reference is usable."""
    if not (0 < threshold < 1 and 0 <= low < high <= 1):
        raise ValueError("Require 0<threshold<1 and 0<=low<high<=1.")
    reference = radiance[1]
    usable = np.all((ldr > low) & (ldr < high), axis=-1)
    reference_usable = usable[1]
    disagreement = np.max(np.abs(radiance - reference[None]) /
                          (radiance + reference[None] + 1e-6), axis=-1)
    flags = {}
    classifications = {}
    for index, name in ((0, "short"), (2, "long")):
        mismatch = usable[index] & (disagreement[index] > threshold)
        weak_or_clipped = ~usable[index]
        flags[name] = reference_usable & (mismatch | weak_or_clipped)
        # 0 retained, 1 disagreement, 2 source unusable, 3 reference unusable.
        classes = np.zeros(reference_usable.shape, dtype=np.uint8)
        classes[reference_usable & mismatch] = 1
        classes[reference_usable & weak_or_clipped] = 2
        classes[~reference_usable] = 3
        classifications[name] = classes
    fallback = flags["short"] | flags["long"]
    prediction = base.weighted_merge(ldr, radiance)
    prediction[fallback] = reference[fallback]
    return prediction, fallback, flags, classifications, reference_usable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Maximum per-channel symmetric relative disagreement; default .2 is provisional, not tuned.")
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output exists. Choose a new name to preserve earlier results.")
    try:
        paths = sorted(p for p in args.scene.iterdir() if p.suffix.lower() in (".tif", ".tiff"))
        exponents = np.loadtxt(args.scene / "exposure.txt").reshape(-1)
        if len(paths) != 3 or exponents.size != 3 or not np.isfinite(exponents).all():
            raise ValueError("Expected three TIFFs and finite exposure exponents.")
        times = np.exp2(exponents)
        if not np.all(np.isfinite(times) & (times > 0)):
            raise ValueError("Invalid exposure values.")
        pairs = sorted(zip(paths, times), key=lambda item: item[1])
        images = []
        for path, _ in pairs:
            x = base.read_rgb(path)
            if not np.issubdtype(x.dtype, np.unsignedinteger):
                raise ValueError("Expected unsigned integer TIFFs.")
            images.append(x.astype(np.float32) / np.iinfo(x.dtype).max)
        ldr = np.stack(images)
        times = np.asarray([t for _, t in pairs], dtype=np.float32)
        radiance = ldr ** 2.2 / times[:, None, None, None]
        prediction, fallback, flags, classifications, reference_usable = reject_motion(ldr, radiance, args.threshold)
        target = base.read_rgb(args.scene / "HDRImg.hdr").astype(np.float32)
        if target.shape != radiance.shape[1:] or not np.all((target >= 0) & (target <= 1)):
            raise ValueError("Target must match dimensions and declared [0,1] convention.")
        height, width = target.shape[:2]
        if min(height, width) < 7:
            raise ValueError("Image too small for 7x7 SSIM.")
        if args.crop:
            x, y, w, h = args.crop
            if min(x, y) < 0 or min(w, h) <= 0 or x + w > width or y + h > height:
                raise ValueError("Crop is outside image.")
        reference = base.reference_only(radiance)
        weighted = base.weighted_merge(ldr, radiance)
        methods = {"reference_only": reference, "weighted_merge_no_alignment": weighted,
                   "reference_fallback_rejection": prediction}
        rows = []
        for name, image in methods.items():
            psnr, ssim = base.evaluate(image, target)
            rows.append({"scene_id": args.scene.name, "method": name, "psnr_mu": psnr, "ssim_mu": ssim})
        args.out.mkdir(parents=True, exist_ok=False)
        with (args.out / "metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        np.save(args.out / "reference_fallback_rejection.npy", prediction)
        np.save(args.out / "reference_fallback_mask.npy", fallback)
        np.save(args.out / "short_rejection_classes.npy", classifications["short"])
        np.save(args.out / "long_rejection_classes.npy", classifications["long"])
        config = {"status": "completed", "scene_id": args.scene.name,
                  "evaluation_status": "Single training-scene development diagnostic, not held-out performance",
                  "platform": platform.platform(), "python": sys.version.split()[0],
                  "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "dependency_sha256": hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
                  "relative_exposure_times": times.tolist(), "gamma": 2.2,
                  "threshold": args.threshold, "encoded_usable_interval": [0.05, 0.95],
                  "disagreement_formula": "max_RGB(abs(H_i-H_ref)/(H_i+H_ref+1e-6))",
                  "fallback_rule": "If every reference RGB channel is inside (0.05,0.95), use reference-only when either source is outside that interval in any channel or disagrees above threshold. Else preserve original weighted merge.",
                  "threshold_status": "Provisional development setting, not validated or test tuned",
                  "reference_usable_fraction": float(reference_usable.mean()),
                  "fallback_fraction": float(fallback.mean()),
                  "mask_class_legend": {"0": "source retained", "1": "radiance disagreement", "2": "source weak/clipped cue", "3": "reference weak/clipped; rejection unavailable"},
                  "mask_limit": "Input-based proxy only, not true motion. Source weakness may come from noise/encoding/exposure; reference saturation can leave ghosts.",
                  "target_usage": "Metric evaluation and display only; not rejection or radiance scaling",
                  "metrics": "Full-image RGB mu-law PSNR and SSIM, mu=5000, peak/data_range=1, uniform SSIM 7x7 with sample covariance. Same conventions as original baseline.",
                  "crop_display_only_xywh": args.crop,
                  "display": "One global Reinhard mapping based on reference only; gamma 1/2.2; original metrics arrays unchanged",
                  "code_attribution": "Supplied starting classical baseline implementation; review and attribute honestly"}
        (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        luminance = reference.astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722])
        average = max(float(np.exp(np.mean(np.log(luminance + 1e-6)))), 1e-6)
        panels = [("Reference only", reference), ("Weighted merge", weighted),
                  ("Reference fallback rejection", prediction), ("Dataset HDR target", target)]
        fig, axes = plt.subplots(2 if args.crop else 1, 4, squeeze=False, figsize=(19, 9 if args.crop else 5))
        for col, (title, image) in enumerate(panels):
            view = base.display_map(image, average)
            scale = min(1, 800 / max(height, width))
            axes[0, col].imshow(np.clip(cv2.resize(view, (round(width*scale), round(height*scale)), interpolation=cv2.INTER_AREA), 0, 1))
            axes[0, col].set_title(title, fontsize=11); axes[0, col].axis("off")
            if args.crop:
                x, y, w, h = args.crop
                axes[0, col].add_patch(plt.Rectangle((x*scale, y*scale), w*scale, h*scale, fill=False, edgecolor="red", linewidth=1.2))
                axes[1, col].imshow(view[y:y+h, x:x+w]); axes[1, col].set_title("Same display crop", fontsize=10); axes[1, col].axis("off")
        fig.suptitle(f"Scene {args.scene.name} / conservative reference fallback / development diagnostic", fontsize=14)
        fig.text(.5,.025,"One common display mapping. Full-image metrics. No learned model or alignment. No target-fitted scaling.",ha="center",fontsize=10)
        fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(args.out / "comparison.png",dpi=150);plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        cmap=ListedColormap(["#f5f5f5", "#c83737", "#e9ad38", "#7b8794"])
        for ax, key in zip(axes[:2], ("short", "long")):
            ax.imshow(classifications[key],cmap=cmap,vmin=0,vmax=3,interpolation="nearest")
            ax.set_title(f"{key.capitalize()} source decision");ax.axis("off")
        axes[2].imshow(fallback,cmap="gray",vmin=0,vmax=1,interpolation="nearest")
        axes[2].set_title("Final fallback mask / white = reference only");axes[2].axis("off")
        fig.suptitle("White: retained | Red: disagreement | Yellow: source weak/clipped | Gray: reference unreliable",fontsize=11)
        fig.text(.5,.03,"These are rule decisions, not ground-truth motion labels. Gray reference regions keep the weighted merge.",ha="center",fontsize=10)
        fig.tight_layout(rect=(0,.08,1,.92));fig.savefig(args.out / "rejection_masks.png",dpi=150);plt.close(fig)
        for row in rows:
            print(f"{row['method']}: PSNR-mu={row['psnr_mu']:.4f} dB, SSIM-mu={row['ssim_mu']:.6f}")
        print(f"Reference usable: {reference_usable.mean():.1%}; reference-only fallback: {fallback.mean():.1%}")
        print(f"Saved comparison.png, rejection_masks.png, metrics.csv and config.json in {args.out}")
        print("Threshold is provisional. Masks are rejection decisions, not ground-truth motion.")
    except (OSError, ValueError, cv2.error) as exc:
        parser.exit(1, f"Rejection baseline failed: {exc}\n")


if __name__ == "__main__":
    main()
