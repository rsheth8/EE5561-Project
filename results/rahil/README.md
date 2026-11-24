# Rahil's completed analysis and frozen test evidence

These are the transferred November 24 records that were missing from the original validation-only repository. Original records and helper code are preserved byte-for-byte. [artifact_sha256.json](artifact_sha256.json) records their packaged hashes. Historical READ_RESULTS files are retained unchanged; current completion status is in [the handoff](../../docs/HANDOFF.md).

| Folder | Scope | Main records |
| --- | --- | --- |
| [scene001](scene001/) | Initial training-scene development diagnostic; not holdout | `metrics.csv`, `config.json`, comparison and rejection masks |
| [validation-regions](validation-regions/) | Five model-selection validation scenes; exploratory analysis | `per_scene_regions.csv`, `summary_regions.csv`, `paired_differences.csv`, `full_image_score_audit.csv`, config and original analysis code |
| [frozen-test](frozen-test/) | All 15 separate test entries, eight methods | `per_scene_metrics.csv`, `summary_metrics.csv`, frozen protocol, manifests, runtime, completion records, all 30 diagnostic figures |
| [audit](audit/) | Saved-record verification and figure provenance | Test/regional verification, report-crop provenance and checked Overleaf export status |

The test has 120 unique method/entry scores and 24 recomputed aggregate groups. See [current results and limits](../TEST_RESULTS.md). Float predictions remain on MSI. The PNGs are display evidence, not metric arrays.

Rahil's whole-pixel fallback differs from repository C2: it uses a `(0.05, 0.95)` all-channel reference usability rule and a 0.20 max-RGB symmetric disagreement threshold. C2 uses the channel-wise rules in the frozen test protocol. Preserve the distinction.

Quick visual review: [test PeopleTalking comparison](frozen-test/figures/PAPER/PeopleTalking/comparison.png), [its error map](frozen-test/figures/PAPER/PeopleTalking/errors.png), and [all-15 paired score changes](frozen-test/M3_minus_M2_all15.png). The report uses a compact layout of the already rendered PeopleTalking panels with the input-selected crop `(240,220,256,256)`; see [provenance](audit/test_figure_provenance.json).
