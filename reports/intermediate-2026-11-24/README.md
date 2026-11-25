# Current intermediate-report snapshot

This is the test-integrated handoff snapshot, November 24, 2026. Both authors must review the writing and actual contributions. Rahil confirms Zijian already has access to the shared Overleaf project with the latest update; continue editing there. No Canvas submission is confirmed.

- [TeamHDR_EE5561_F26.pdf](TeamHDR_EE5561_F26.pdf): compiled source snapshot, five content pages plus one reference page.
- [overleaf-export-checked.pdf](overleaf-export-checked.pdf): historical shared Overleaf export from before the final wording/notation changes. It was checked against its earlier snapshot and is retained for provenance, not as the current submission PDF. A fresh live export must be checked after the authors finish editing.
- [main.tex](main.tex) and [refs.bib](refs.bib): editable report and bibliography.
- [Overleaf_update_files.zip](Overleaf_update_files.zip): only the seven root source/support files and three figures needed for Overleaf, without the large evidence folders.

## Final text changes

The current `main.tex` uses the cautious cue conclusion, clearer cue intuition and an ordered scientific plan. It defines $Z_2$, $Z_i^o$ and $Z_i^w$, clarifies the per-channel fallback in C1-C3, and removes the standalone Assistance paragraph and colored link boxes. The template support files, figures and numerical results are unchanged. The local source was compiled and all six rendered pages were checked.

## Editing and compiling

Use the existing shared Overleaf project for team editing. Preserve any newer collaborator changes. Keep `main.tex`, `refs.bib`, `style.sty`, `refs.bst`, `natbib.sty`, `fancyhdr.sty`, and `latexmkrc` at the project root, with the three PNGs under `figures/`. All five template support files are unchanged from the supplied F26 template. The report has no dependency on `results/rahil/` or audit folders for compilation.

Choose `main.tex` as the main document. XeLaTeX plus BibTeX works; the checked shared export also compiled correctly with pdfLaTeX. Locally, from this folder:

```bash
tectonic --keep-logs main.tex
```

Re-render and inspect every page after changes. The limit is five content pages excluding references. Download one final PDF named `TeamHDR_EE5561_F26.pdf`; agree which member submits it and verify the actual Canvas deadline/upload.

## Current figure/table numbering

- Table 1: training/validation/test partitions.
- Table 2: validation PSNR-mu, test PSNR-mu and test SSIM-mu for all eight methods.
- Figure 1: architecture schematic.
- Figure 2: saved validation curves.
- Figure 3: input-selected PeopleTalking face comparison, common display mapping, subtle M0-M3 differences.

Regional teaching guides and historical evidence notes can use older figure numbering. They are not the current report. See [the handoff](../../docs/HANDOFF.md) for required human review.
