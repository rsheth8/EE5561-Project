# Current validation and frozen test results

Updated November 24, 2026. This supplements the original generated [validation summary](RESULTS.md). The current [report](../reports/intermediate-2026-11-24/README.md) includes the separate test results.

## All 15 test entries

Source: [per_scene_metrics.csv](rahil/frozen-test/per_scene_metrics.csv) and [summary_metrics.csv](rahil/frozen-test/summary_metrics.csv). C0-C3 are classical controls; M0-M3 are the frozen learned variants. PSNR is in dB; all rows use equal-entry means and the same test runtime.

| Method | Description | Test PSNR-mu | Test SSIM-mu |
| --- | --- | ---: | ---: |
| C0 | Reference only | 38.188737 | 0.975273 |
| C1 | Weighted merge | 24.901399 | 0.830546 |
| C2 | Channel-wise rejection | 28.314809 | 0.922842 |
| C3 | Aligned rejection | 35.087431 | 0.969033 |
| M0 | Original inputs | 42.902696 | 0.989568 |
| M1 | Aligned inputs | 42.821191 | 0.990361 |
| M2 | Dual path, constant cues | 43.302421 | 0.990460 |
| M3 | Dual path, measured cues | 43.388039 | 0.990517 |

M3 minus M2 is +0.085617 dB, with nine wins and six losses. PAPER mean difference is +0.244949 dB and EXTRA is +0.005952 dB. ManStanding contributes +1.260938 dB. A post-hoc sensitivity check excluding it gives +0.001666 dB across the other 14; all 15 remain in the primary result. See [all paired changes](rahil/frozen-test/paired_M3_minus_M2_audited.csv) and [their plot](rahil/frozen-test/M3_minus_M2_all15.png).

## Validation and meaning of the controls

The [saved validation table](tables/variants_val.csv) gives M2 44.1342 and M3 44.0910 dB PSNR-mu: the direct cue effect is slightly negative (-0.0432 dB). Each pilot used seed 5561, 50,000 updates, and validation-selected checkpoints, all at step 50,000. Training scene count is 69; validation IDs are 025, 045, 051, 056, 062. Test is a separate archive with five PAPER and ten EXTRA entries.

M0 uses original inputs; M1 uses aligned inputs with the same backbone. M2 adds an original/aligned path and gate with constant cue channels. M3 has equal capacity and measured exposure/validity/forward-backward consistency cues. Both M2 and M3 keep the hard warp-validity mask. M3 versus M2 tests the cues; M2 versus M1 also changes capacity and architecture.

Test alignment is not uniformly beneficial: C3 scores 35.087431 versus C0 38.188737 dB; M1 is 0.081505 dB below M0. M3's small mean gain and one seed do not establish a general or statistically significant advantage.

## Rahil's regional validation analysis

The [regional records](rahil/validation-regions/) and [audit](rahil/audit/received_regional_analysis_verification.json) show all 20 re-scored model/validation pairs match saved full-image metrics to numerical precision. Bright-reference mu-MAE improves with M3 only for scene 062 (7.84%); it worsens for the other four. Scene 045 has only 24 bright pixels. These brightness masks are input proxies, not verified motion masks, and these are model-selection scenes, not independent holdout evidence. Rahil's whole-pixel fallback is distinct from C2.

## Verification scope and limits

[Test audit](rahil/audit/received_test_results_verification.json): 120 complete unique method/entry records; 24 aggregate groups recomputed with maximum roundoff 1.42e-14; frozen protocol and manifest hashes match. Input hashes and reference thumbnails checked against all 74 Training scenes reported no overlap/correlation flags; undetected relatives remain possible.

EXTRA/003 and PAPER/LadySitting share a reference and target, as do EXTRA/005 and PAPER/PeopleStanding. Keep them in the published 15-entry mean but disclose that entries are correlated. Full float test arrays remain on MSI, so the transferred-record audit is not independent rescoring of all predictions. Checkpoint bytes were absent from the transfer; the evaluator checks frozen hashes, and `scripts/verify_handoff.py` compares them with the already released repository copies.

Training/validation used torch 2.7.1+cu126 and OpenCV 5.1.0; test used torch 2.7.1+cu118 and OpenCV 4.10.0. Compare methods within each partition. The test runtime was 271.365 seconds. Do not attribute validation-to-test changes solely to data or claim Challenge123/camera-shake/cross-dataset results.

## Figure reading

The [PeopleTalking report figure](../reports/intermediate-2026-11-24/figures/test_people_talking.png) uses the input-selected face crop `(240,220,256,256)`. C1 doubles the face and C3 leaves artifacts; M0-M3 look similar. Outputs and target share the same reference-derived Reinhard exposure/gamma display. Numeric metrics use original float arrays and a fixed mu mapping. [Figure provenance](rahil/audit/test_figure_provenance.json) records the panel rearrangement; no color/contrast manipulation was applied.
