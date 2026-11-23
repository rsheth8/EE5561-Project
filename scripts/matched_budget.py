"""6d matched-budget rule.

python scripts/matched_budget.py select      -> configs/variants-eval.json (N* = largest eval step all 4 reached)
python scripts/matched_budget.py table <eval_run>  -> results/tables/variants_val.csv, variants_diffs.csv, fig_variants_curve.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hdr.display import write_sidecar  # noqa: E402
from hdr.evaluate import resolve_run  # noqa: E402
from hdr.metrics import METRIC_KEYS  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
VARIANTS = ["M0", "M1", "M2", "M3"]
LABEL = "preliminary, single seed, 5 validation scenes"


def val_logs():
    out = {}
    for v in VARIANTS:
        try:
            rd = resolve_run({"exp": f"pilot_{v}", "run_dir": "latest"})
        except (FileNotFoundError, StopIteration) as exc:
            out[v] = (None, None, str(exc))
            continue
        vl = pd.read_csv(rd / "val_log.csv", dtype={"scene": str}).drop_duplicates(["step", "scene"], keep="last")
        out[v] = (rd, vl, None)
    return out


def select():
    logs = val_logs()
    steps = {v: set(vl[vl.scene == "mean"].step) for v, (rd, vl, e) in logs.items() if vl is not None}
    missing = [v for v in VARIANTS if v not in steps]
    common = set.intersection(*steps.values()) if steps and not missing else set()
    nstar = max(common) if common else None
    info = {"N_star": nstar, "eval_steps_reached": {v: sorted(int(s) for s in st) for v, st in steps.items()},
            "max_step_reached": {v: int(max(st)) for v, st in steps.items()}, "missing": missing,
            "run_dirs": {v: str(rd.relative_to(REPO)) for v, (rd, vl, e) in logs.items() if rd is not None}}
    entries = []
    if nstar is not None:
        for v in VARIANTS:
            entries.append({"name": v, "variant": v, "exp": f"pilot_{v}", "run_dir": info["run_dirs"][v], "max_step": nstar})
    cfg = {"split": "configs/split.json", "scenes": "val", "entries": entries, "save_outputs": True, "matched_budget": info}
    (REPO / "configs" / "variants-eval.json").write_text(json.dumps(cfg, indent=2))
    (REPO / "results" / "tables" / "matched_budget.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=1))


def table(eval_run: str):
    ev = Path(eval_run).resolve()
    m = pd.read_csv(ev / "metrics.csv", dtype={"scene": str})
    mj = json.loads((ev / "metrics.json").read_text())
    info = json.loads((REPO / "results" / "tables" / "matched_budget.json").read_text())
    nstar = info["N_star"]
    sel = {k: v["step"] for k, v in mj["selection"].items()}
    m = m.rename(columns={"name": "model"})[["scene", "model", "step"] + METRIC_KEYS]
    hdr = (f"# {LABEL}; full image, mu = 5000; matched-budget rule N* = {nstar} (largest eval step all four variants reached); "
           f"each variant's checkpoint = best val PSNR-mu among steps <= N*: {sel}; source run {ev.relative_to(REPO)}\n")
    with (REPO / "results" / "tables" / "variants_val.csv").open("w") as fh:
        fh.write(hdr)
        m.sort_values(["scene", "model"]).to_csv(fh, index=False)
    piv = m.set_index(["scene", "model"])[METRIC_KEYS]
    rows = []
    for a, b, factor in (("M1", "M0", "alignment"), ("M2", "M1", "extra path + gate"), ("M3", "M2", "explicit cues")):
        for sid in sorted(m.scene.unique()):
            d = piv.loc[(sid, a)] - piv.loc[(sid, b)]
            rows.append({"diff": f"{a}-{b}", "factor": factor, "scene": sid, **d.to_dict()})
    dd = pd.DataFrame(rows)
    with (REPO / "results" / "tables" / "variants_diffs.csv").open("w") as fh:
        fh.write(hdr)
        dd.to_csv(fh, index=False)
    # curves
    logs = val_logs()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for v, (rd, vl, e) in logs.items():
        if vl is None:
            continue
        vm = vl[vl.scene == "mean"].sort_values("step")
        ax.plot(vm.step, vm.psnr_mu, "o-", ms=3, label=f"{v} (selected step {sel.get(v)})")
    if nstar:
        ax.axvline(nstar, color="k", ls=":", label=f"N* = {nstar}")
    ax.set_xlabel("update")
    ax.set_ylabel("val PSNR-mu (dB), mean of 5 scenes")
    ax.legend(fontsize=8)
    ax.set_title(LABEL)
    fig.tight_layout()
    fp = REPO / "results" / "figures" / "fig_variants_curve.png"
    fig.savefig(fp, dpi=100)
    plt.close(fig)
    write_sidecar(fp, [f"Validation PSNR-mu (5 val scenes, full image, mu=5000) vs updates for M0-M3; {LABEL}.",
                       f"Matched-budget N* = {nstar}; selected steps {sel}. Sources: runs/pilot_M*/<ts>/val_log.csv, {ev.relative_to(REPO)}."])
    print(m[m.scene == "mean"].to_string())
    print(dd[dd.scene == "mean"].to_string())


if __name__ == "__main__":
    if sys.argv[1] == "select":
        select()
    else:
        table(sys.argv[2])
