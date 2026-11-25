# EE5561 HDR agent context

Read `docs/HANDOFF.md`, `results/TEST_RESULTS.md` and `docs/REPRODUCING.md` before work. The current intermediate report backup is `reports/intermediate-2026-11-24/main.tex`. Rahil confirms the shared Overleaf project has the latest update and Zijian already has access. Continue author edits there; never replace newer shared edits with an older repository snapshot. Submit as `TeamHDR_EE5561_F26.pdf`.

## Preserve the experiment

- The reviewed implementation baseline is commit `9f83e01ec1d940e6255bd801c7559434c0b8e6ef`. Handoff additions do not change model/training methods or frozen checkpoints.
- Do not overwrite original `runs/`, released checkpoints, split/configuration files or transferred evidence while editing prose or organizing documentation.
- The 69/5 training/validation split is fixed. Test is the separate 15-entry archive. Keep all 15 in the primary equal-entry average; disclose duplicate reference/target pairs.
- Test records are verified, with aggregates recomputed. Full float test arrays were not transferred. Do not describe this audit as independent rescoring of all 120 predictions.
- The frozen evaluator checks checkpoint hashes. Do not substitute weights, tune on these test scores or change settings while presenting the existing results. New experiments need an explicit student decision and separate output directories.
- Use `uv` for Python. Heavy training/evaluation belongs in an allocated MSI GPU session, not a login node or an agent's local shell.

## Interpretation

- M0: original inputs. M1: aligned inputs. M2: dual path and constant cues. M3: same capacity with measured cues. M2 and M3 both retain the hard warp-validity mask.
- M3 minus M2 is -0.0432 dB validation PSNR-mu and +0.0856 dB test PSNR-mu. Test is 9 wins/6 losses. The post-hoc other-14 mean excluding ManStanding is only +0.0017 dB; all 15 stay primary.
- One seed and correlated test entries do not establish statistical significance, general robustness or cross-dataset performance.
- Training/validation used torch cu126 and OpenCV 5.1.0; test used torch cu118 and OpenCV 4.10.0. Compare methods within partitions; do not attribute validation-to-test changes solely to data.
- Rahil's whole-pixel fallback is not repository C2. Regional brightness flags are proxies, not verified motion masks. Scene 045 has only 24 bright pixels. Regional analysis reuses validation scenes.
- Do not invent visible M3 improvements in the report crop. Preserve common display mapping and input-selected crop provenance.

## Writing and ownership

- R/Z guide folder names reflect the old plan, not proven implementation authorship. Use the observed contributions in the handoff and have both students confirm them.
- All report prose remains an author-review draft. Zijian must check the whole report and rewrite explanations of his methods, training and analysis in his own voice and understanding. Rahil must confirm the account of his work.
- Agents may help check facts and clarity, but must not fabricate student authorship, comprehension or approval. Rewriting means explaining the work accurately, not disguising assistance.
- Rahil requested removal of the standalone Assistance paragraph from the report. Do not automatically reinsert it. Keep accurate recorded provenance and completed/future-work distinctions. Challenge123, camera-shake evaluation, individual-cue ablations, extra seeds and cross-dataset claims are not completed results.
- No Canvas submission, email or other message is authorized merely by this file. Confirm submission status with the students.

## Checks

Run `uv run scripts/verify_handoff.py` for documentation/evidence changes. Compile the report after LaTeX edits and inspect all pages: maximum five content pages excluding references. See `docs/REPRODUCING.md` for the existing test suite and runtime limits. Use plain, specific writing without em dashes.
