# Code map

The implementation remains in `GeoStep/` to preserve original filenames and local imports. Research scripts and protocols are copied byte-for-byte from the archived source package; see `provenance/source_manifest.json`.

## Start with the formal implementation

| Purpose | Files |
|---|---|
| 14.99M backbone | `capacity15_models.py`, with inherited blocks/forward logic in `pilot.py` |
| Paired mask generation | `matched_masks.py`, `matched_mask_protocol.json` |
| Common FM training/evaluation | `train_matched.py`, `followup.py`, `continue30k.py` |
| Final FM training | `train_capacity15.py`, `train_capacity15_seed1.py` |
| Pixel regression | `train_regression15.py`, `train_regression15_seed1.py`, `regression15_eval.py` |
| Perceptual regression | `perceptual15_loss.py`, `train_perceptual15.py`, `perceptual15_pipeline.py` |
| Original test protocol | `official_test_v9.py`, `official_test_protocol_v9.json`, `seed1_pipeline.py` |
| Capacity diagnostics | `capacity_models.py`, `train_wide.py`, `compare_capacity.py`, `compare_capacity15.py` |
| Provenance/consistency checks | `preflight_*.py`, `verify_*.py` |

## Why early-stage files remain

`pilot.py` supplies network blocks, data preparation and sampling; `followup.py` supplies evaluation math; `continue30k.py` supplies loss probes. Later formal scripts import them and check their source hashes. Removing or rewriting those files would change the archived dependency graph. Their early training entry points and legacy mask designs are not the paper's final protocol.

Likewise, `protocol.json` describes the 2k pilot. The paper settings are specified in `capacity15_*`, `regression15_*`, `official_test_protocol_v9.json` and `perceptual15_protocol_v10.json`. The repository excludes the separate `pmrf_spatial_noise` exploration because it is not part of the manuscript's evidence.

## Results and public entry points

Use `scripts/show_results.py` and `scripts/plot_results.py` for the published result package. Historical `analyze_*.py` and verification scripts expect full run directories not included here. They are retained for inspection, not advertised as fresh-clone commands.
