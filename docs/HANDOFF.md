# Handoff to Zijian

Updated November 24, 2026. This supersedes the status in the November 23 step-by-step guide. It is a catch-up record and task list, not a claim that both authors have approved the draft.

[Companion Google Doc](https://docs.google.com/document/d/1_P_xWd0WOt38PF-fTKYo3yZAXpBfYsVtzTglldpVIQU/edit) is shared with Zijian for editing. This repository file is the versioned agent reference.

## Read this first

The experiments needed for the intermediate-report draft are complete. Rahil has integrated his MSI checks, classical/regional analysis and the separate frozen test evaluation into the required LaTeX template. He manually updated the shared Overleaf project and sent its exported PDF back for checking. That earlier export was checked against its then-current snapshot. Since then, the source received the final wording and notation fixes below. The current source compiles to five content pages plus references; no fresh live Overleaf export has been independently checked after these edits.

**Zijian's immediate job is to review and own the writing before submission.** Read the whole report, check the methods and claims against the records, and rewrite explanations of your implementation, training and reasoning in your own voice and understanding. Do more than grammar cleanup. You should be able to explain each equation, baseline, comparison and limitation without relying on agent wording. Rahil should confirm the account of his work and both authors should confirm the contribution statements.

The intermediate instructions say one PDF per group, maximum five content pages excluding references, due November 24. The instructions do not give an exact time. Check Canvas. Nothing in this handoff confirms a Canvas submission.

## Latest report revision and where to work

Rahil confirms that Zijian already has access to the shared Overleaf project with the latest update. Continue the final writing pass there. GitHub is the matching backup and evidence/agent reference; do not upload an older ZIP over newer shared edits.

The revised source strengthens the cue conclusion, explains the cues in plain language, connects scientific problems to ordered next steps, defines the feature symbols near Equation 3, and specifies that the low-weight fallback applies per channel in C1-C3. The standalone Assistance paragraph and colored hyperlink boxes were removed at Rahil's request. Results, methods, checkpoints, the supplied template support files and all three figures are unchanged.

The compiled backup is `reports/intermediate-2026-11-24/TeamHDR_EE5561_F26.pdf`. Recompile the shared project and inspect its actual export after your edits. The earlier `overleaf-export-checked.pdf` is retained only as a historical checked export.

## What to do before submitting

1. Open the existing shared Overleaf project and read the current draft, including captions, references and contributions. Its live URL is not recorded in this repository; use the shared project already used by the team. Avoid replacing newer collaborator edits with the saved snapshot.
2. Check technical accuracy. Explain gamma/exposure conversion, C0-C3, DIS alignment, attention/dilation, M0-M3 and the equal-capacity M2/M3 cue control in words you understand. Check the shift from the original proposal.
3. Rewrite unclear or agent-like passages yourself. Focus especially on your model/training work, related-work positioning, interpretation and future plan. Preserve factual limits and original metrics. Review Rahil's sections too, while keeping his actual contribution accurate.
4. Confirm names/emails and what each person actually did. Folder labels do not establish authorship.
5. Recompile and inspect all pages after edits. The current snapshot is five content pages plus references; Overleaf/compiler or wording changes can move material onto a sixth content page.
6. Agree who submits, confirm the exact Canvas deadline and whether a submission already exists, then have one member upload the final PDF as `TeamHDR_EE5561_F26.pdf`. Check the uploaded/downloaded file.

## What Rahil completed

- Proposed the photography/ghosting motivation and verified his MSI OnDemand GPU environment and parameter updates.
- Downloaded/inventoried data and inspected scene 001. The initial whole-pixel rejection diagnostic is a development example on a training scene, not held-out evidence.
- Ran classical comparisons and regional failure analysis across validation scenes 025, 045, 051, 056 and 062. Re-scored frozen M0-M3 predictions; all 20 full-image PSNR/SSIM pairs matched saved records to numerical precision. Checked region counts and arithmetic, including the 24-pixel bright region in scene 045.
- Ran the frozen C0-C3 and M0-M3 evaluation on all 15 separate test entries on an MSI A100. Preparation compared test inputs with all 74 Training-archive scenes; saved records report no exact overlap or reference-thumbnail correlation flags. This cannot prove every related capture was detected.
- Transferred results, inspected diagnostic figures, and integrated the test results, clear face crop, runtime differences and weak cue findings into the report.
- Updated the shared Overleaf project manually. Its earlier exported PDF was checked against the earlier snapshot: matching text/figures, five content pages plus references, no observed clipping or overlaps. The current source has since received the final text revisions.

Rahil used AI-supported starting scripts, figures, checking and drafting. Do not turn execution/analysis work into a claim that he independently authored every supplied implementation.

## What Zijian completed

- Implemented the shared data/model pipeline, flow alignment, reliability cues and original/aligned gate.
- Ran M0/M3 fitting checks and four model pilots, each with seed 5561 and 50,000 updates. Validation was evaluated every 2,000 updates; all four selected checkpoints are step 50,000 within the common budget.
- Organized original run records, checkpoints, validation comparisons and figures.

These contribution descriptions reflect observed work and still need the authors' confirmation. The R-labeled original folders include provisional work supplied so the model pipeline could run; they are not evidence that Rahil wrote that code. His later analysis/test records are separately stored under `results/rahil/`.

## What the experiments say

See [current results](../results/TEST_RESULTS.md) and the linked CSV/JSON records for exact values.

| Comparison | Validation PSNR-mu | Test PSNR-mu |
| --- | ---: | ---: |
| M2 constant cues | 44.1342 | 43.3024 |
| M3 measured cues | 44.0910 | 43.3880 |
| M3 minus M2 | -0.0432 | +0.0856 |

On test, M3 wins nine entries and loses six. PAPER mean change is +0.244949 dB; EXTRA is +0.005952 dB. ManStanding contributes +1.260938 dB. Removing it as a post-hoc sensitivity check leaves only +0.001666 dB across the other 14. Keep all 15 in the primary result. The report crop shows classical ghosting clearly, but M0-M3 look similar; do not claim a clear visible M3 win.

The test contains two pairs with identical reference/target files: EXTRA/003 with PAPER/LadySitting, and EXTRA/005 with PAPER/PeopleStanding. They remain benchmark entries, not 15 established independent capture conditions. With one seed, this does not demonstrate a reliable general cue advantage, statistical significance or cross-dataset robustness.

All 120 saved method/entry records are complete and unique, and all 24 aggregate groups were recomputed (maximum difference 1.42e-14). This is saved-record verification. Full float test arrays remain on MSI and were not independently rescored locally. The original transfer did not contain checkpoint bytes; the evaluator enforces frozen hashes. The repository verifier additionally compares those hashes with the checkpoint copies already released in this repo.

## Changes from the proposal and feedback

The original proposal emphasized a synthetic motion benchmark, motion/attention ablations and a photo-only method selector. The current work instead emphasizes captured-scene controls and an original/aligned feature gate with a measured-cue control. The report now states the change. Do not present the proposed benchmark, selector or augmentation runs as completed.

The instructor requested explicit baselines, gamma, attention and dilation; a clear contribution beyond AHDRNet; a definition of shake; and less extensive LLM writing. The TA asked for stronger workload/implementation ownership and captured-motion evaluation such as Challenge123. The report defines the current methods and acknowledges prior reliability-aware fusion (FlexHDR/SAFNet). Synthetic shifts are code diagnostics only. Challenge123 remains planned, not completed.

## Files and access

- [Report folder](../reports/intermediate-2026-11-24/README.md): source, compiled snapshot, checked Overleaf export and small update ZIP. The saved source is a handoff snapshot, not a replacement for newer shared Overleaf edits.
- [Rahil evidence index](../results/rahil/README.md): original transferred records, figures and audit summaries.
- [Reproduction notes](REPRODUCING.md): environments, original MSI paths and verification commands.
- [Original validation summary](../results/RESULTS.md): original model-pilot evidence. It predates the frozen test update.
- [Original shared guide](https://docs.google.com/document/d/1IcNYyowFOmfJMPCiuJM0EDQDiJTFz5oz0yHjuAqIFJw/edit): historical plan, superseded where this handoff records completed work or corrected ownership.

Raw datasets, caches and float predictions are not added by this update. Do not compute PSNR from display PNGs. Historical evidence notes are preserved unchanged, so some still say to run steps that are already complete. Follow this handoff and current results for status.

## After the intermediate submission

These are proposed next steps, not instructions to launch costly work without discussing scope:

- Zijian: separate exposure-cue and flow-cue ablations and additional seeds; define the comparison and validation-selection rule before running.
- Rahil's proposed follow-up: audit Challenge123 exposure/target conventions and annotate motion/clipping cases. If Rahil is unavailable, agree who takes this on rather than claiming completion.
- Team: decide the feasible external captured-motion and camera-shake evaluation. Camera shake should mean a defined whole-frame motion protocol, distinct from local object motion. A synthetic warp sanity check is not that benchmark.
- Keep today's checkpoints/results frozen. If the existing test informs new tuning, acknowledge that it has become development evidence and plan fresh final evaluation.

## Starting a new agent session

Give the agent the repository and ask it to read `AGENTS.md`, this handoff, `results/TEST_RESULTS.md`, `docs/REPRODUCING.md` and the report source before changing anything. Its first task should be to summarize completed work, current limitations and the specific writing or experiment task you authorize. It must not invent author approval, reuse R/Z labels as credit, restart training just to understand status, or claim unrun future experiments are complete.
