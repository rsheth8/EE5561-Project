# Current intermediate-report snapshot

This is the test-integrated handoff snapshot, November 24, 2026. Both authors must review the writing, actual contributions and assistance disclosure. No Canvas submission is confirmed.

- [HDR_EE5561_F26.pdf](HDR_EE5561_F26.pdf): compiled source snapshot, five content pages plus one reference page.
- [overleaf-export-checked.pdf](overleaf-export-checked.pdf): Rahil's exported shared Overleaf version. Text and all three embedded figures match the source snapshot; all six pages were visually checked. Its pdfLaTeX rendering is valid even though the local snapshot used Tectonic/XeTeX.
- [main.tex](main.tex) and [refs.bib](refs.bib): editable report and bibliography.
- [Overleaf_update_files.zip](Overleaf_update_files.zip): only the seven root source/support files and three figures needed for Overleaf, without the large evidence folders.

## Editing and compiling

Use the existing shared Overleaf project for team editing. Preserve any newer collaborator changes. Keep `main.tex`, `refs.bib`, `style.sty`, `refs.bst`, `natbib.sty`, `fancyhdr.sty`, and `latexmkrc` at the project root, with the three PNGs under `figures/`. All five template support files are unchanged from the supplied F26 template. The report has no dependency on `results/rahil/` or audit folders for compilation.

Choose `main.tex` as the main document. XeLaTeX plus BibTeX works; the checked shared export also compiled correctly with pdfLaTeX. Locally, from this folder:

```bash
tectonic --keep-logs main.tex
```

Re-render and inspect every page after changes. The limit is five content pages excluding references. Download one final PDF named `HDR_EE5561_F26.pdf`; agree which member submits it and verify the actual Canvas deadline/upload.

## Current figure/table numbering

- Table 1: training/validation/test partitions.
- Table 2: validation PSNR-mu, test PSNR-mu and test SSIM-mu for all eight methods.
- Figure 1: architecture schematic.
- Figure 2: saved validation curves.
- Figure 3: input-selected PeopleTalking face comparison, common display mapping, subtle M0-M3 differences.

Regional teaching guides and historical evidence notes can use older figure numbering. They are not the current report. See [the handoff](../../docs/HANDOFF.md) for required human review.
