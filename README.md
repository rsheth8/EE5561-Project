# EE5561 Team HDR

Rahil Sheth and Zijian Zhang, University of Minnesota, Fall 2026.
We study ghosting when merging bracketed exposures, using classical controls and an AHDRNet-style original/aligned feature gate.

## Start here

1. [Handoff to Zijian](docs/HANDOFF.md): completed work, writing review, submission steps and future experiments.
2. [Current results](results/TEST_RESULTS.md): validation, frozen test evaluation and limits.
3. [Intermediate report](reports/intermediate-2026-11-24/README.md): current PDF, LaTeX source and small Overleaf update package.
4. [Reproduction and verification](docs/REPRODUCING.md): saved-record audit, environments and MSI locations.
5. [Agent instructions](AGENTS.md): read before editing or interpreting results.

**Status on November 24, 2026:** four 50,000-update pilots, five-scene validation analysis and all 15 frozen test entries are complete. The report export has five content pages plus references. Both authors still need to review the prose, contributions and assistance disclosure. Canvas submission has not been confirmed.

The direct cue comparison is mixed: M3 is 0.0432 dB below M2 on validation and 0.0856 dB above M2 on test. The test gain is concentrated in ManStanding. This does not establish a general cue advantage.

[Google Docs handoff for Zijian](https://docs.google.com/document/d/1_P_xWd0WOt38PF-fTKYo3yZAXpBfYsVtzTglldpVIQU/edit) accompanies the repository guide and is shared with him for editing.

## Repository map

| Location | Contents |
| --- | --- |
| `hdr/`, `configs/`, `scripts/`, `launch/`, `tests/` | Original model, training, evaluation and checks |
| `runs/`, `results/tables/`, `results/figures/` | Original training and validation records |
| [results/RESULTS.md](results/RESULTS.md) | Generated historical validation summary; current test summary is linked above |
| [results/rahil/](results/rahil/README.md) | Rahil's scene-001 diagnostic, regional analysis and frozen test transfer |
| [tools/frozen-test/](tools/frozen-test/README.md) | Byte-preserved evaluation helper and frozen protocol |
| [release/](release/README.md) | Original released checkpoints and validation display images |
| [results/by_step/](results/by_step/README.md) | Historical guide index; folder labels are not proof of authorship |
| [notes/problems.md](notes/problems.md) | Original problems log |

No raw dataset, cache or full float prediction arrays are added by the handoff. Obtain data from the [Kalantari project page](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/). Data and full predictions remain on MSI.

## Verify the handoff without a GPU

```bash
uv run scripts/verify_handoff.py
```

This checks artifact hashes, frozen helper/protocol consistency, released checkpoint hashes, the complete 120-score test grid, all 24 summary groups and local Markdown links. It does not rescore float predictions or retrain models. See [reproduction notes](docs/REPRODUCING.md) for GPU procedures and the distinction between recorded environments.
