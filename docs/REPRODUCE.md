# Reproduction guide

## What works from a fresh clone

1. `python scripts/check_release.py` checks the source/data manifest, Python syntax, local imports, split counts, and key aggregate-result relationships. It uses the standard library and writes no experiment artifacts.
2. `python scripts/show_results.py` prints the included test summaries and measured inference costs. It does not recompute metrics from images.
3. `python scripts/plot_results.py` redraws the NFE and training-dynamics figures from archived CSV tables. It needs NumPy and Matplotlib, but no model weights or GPU.
4. `python -m unittest discover -s tests -v` runs synthetic CPU implementation checks. It needs the model dependencies and does not download pretrained weights.

## What is not included

Original images and annotations, prepared image caches, matched mask banks, trained checkpoints, most historical initialization/intermediate checkpoints, complete training logs, and full per-image prediction arrays are not shipped. The final training and evaluation entry points therefore require additional historical artifacts. The repository does not claim that the complete historical experiment can be recreated from this checkout alone.

## Historical execution layout

The original scripts were written for a Linux CUDA workspace rooted at `/data/GeoStep`. Several entry points accept `--root`; others fix that path internally. The shared annotation reader expects the Oxford-IIIT Pet dataset at `/data/lgfnet_data/oxford-iiit-pet`. The original split manifest path is `/data/LGFNet/manifests/splits.csv`.

The released `data/splits.csv` matches the SHA-256 expected by the frozen test protocol. `GeoStep/protocol.json` is the early data-preparation/pilot configuration, not the final 30k experimental specification. Its dataset paths remain historical defaults. The public scripts and protocols have not been rewritten or had their hash checks removed.

## Formal entry points and prerequisites

| Stage | Entry point | Prerequisites beyond this repository |
|---|---|---|
| Prepare image cache | `pilot.py --root ROOT --prepare` | Dataset images and split manifest at configured paths |
| Build paired masks | `matched_masks.py --root ROOT` | Prepared data, trimaps/head annotations, and mask protocol |
| 14.99M FM, seed 0 | `train_capacity15.py --root ROOT --seed 0` | v4 baseline records/checkpoints, v3 identity, mask bank and cache |
| 14.99M FM, seed 1 | `train_capacity15_seed1.py --root ROOT --seed 1` | Corresponding seed-1 baseline and frozen data/initialization records |
| Pixel regression | `train_regression15.py` / `train_regression15_seed1.py` | Matched FM initialization and historical verification records |
| Perceptual regression | `train_perceptual15.py --root ROOT --seed 0 --alpha-tag alpha100` | Selected protocol, calibration, matching initialization, VGG16 weights, and prior frozen results |
| Initial test evaluation | `official_test_v9.py` | Frozen FM/pixel checkpoints, test cache or preparation inputs, LPIPS weights and original weight identities |
| Perceptual control pipeline | `perceptual15_pipeline.py` | Completed original test artifacts, preflight/calibration, and training prerequisites |

The table is a dependency map, not a sequence that can be run on a fresh clone. Inspect each entry point and the associated protocol before execution. Some pipelines launch training and subsequently evaluate the test set. The `preflight` label is not a universal guarantee of zero optimizer updates; the legacy pilot preflight performs limited updates.

The trainers preserve the original data streams and compare them against prior run records. They also verify source, protocol, cache, and checkpoint hashes. Replacing missing artifacts with placeholders or editing identities to bypass assertions would no longer reproduce the archived protocol.

## Starting a new experiment

Use a separate workspace, preserve the published source archive, and create a new protocol and provenance record for intentional changes. Porting the historical pipeline to arbitrary paths or removing predecessor dependencies is a separate engineering task, not an already-validated feature of this release. New runs should not be presented as byte-identical reproductions of the archived experiment unless that equivalence is checked.

## Metric and timing interpretation

- Convert image values from `[-1, 1]` to `[0, 1]` before computing missing-region MSE and PSNR.
- LPIPS uses calibrated SqueezeNet features on each full composed image at native 64 × 64 resolution.
- Score each FM sample separately, then average four noise samples and mask conditions. Do not average images before scoring or select the best sample.
- The timing records measure one output per image, with GPU-resident inputs, five warm-ups and twenty synchronized repetitions. I/O and metric computation are excluded. Batch-64 milliseconds per image describe throughput rather than single-image latency.
