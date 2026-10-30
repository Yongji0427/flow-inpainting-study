# One Flow, Two Notions of Quality

### Sampling Budgets for Image Inpainting

**Yongji Chen · Kaichun You · Linwei Zhang · Zijun Tang**

Research code and archived results for a controlled comparison of conditional flow matching, pixel regression, and perceptual regression on **64 × 64 Oxford-IIIT Pet** images.

The study asks how much additional sampling computation helps when the trained model is fixed. Across the tested development budgets, **2 function evaluations give the highest masked PSNR, while 8 give the lowest LPIPS**. Larger budgets do not keep improving these reference-based metrics.

![Development quality versus sampling budget](figures/nfe_quality.png)

*Each curve uses frozen checkpoints and fixed image–mask–noise conditions. Thin lines show the two training seeds; thick lines show their mean. The two mask geometries are reported separately. These are development-set results, not a test-set NFE sweep.*

## Start here

These commands inspect the released files and display archived results. They require only Python 3.10+ and do not train a model, run inference, or download weights.

```bash
git clone https://github.com/Yongji0427/flow-inpainting-study.git
cd flow-inpainting-study
python scripts/check_release.py
python scripts/show_results.py
```

To redraw the numerical figures from the included CSV tables:

```bash
python -m pip install -r requirements-figures.txt
python scripts/plot_results.py
```

**Reproduction scope.** The original research scripts are preserved byte-for-byte. Training and test pipelines depend on historical initialization files, prepared data, mask banks, and frozen provenance records that are not all included here. A fresh clone supports code inspection, aggregate-result checks, and numerical figure regeneration. It does **not** support a one-command replay of the complete training history. See [Reproduction guide](docs/REPRODUCE.md) before running the historical entry points.

## Controlled comparison

| Model | Training objective | Inference |
|---|---|---|
| Pixel regression | Missing-region image MSE | One CNN evaluation |
| Perceptual regression | Image MSE + frozen VGG16 feature loss | One CNN evaluation; no VGG pass at inference |
| Conditional flow matching | Missing-region velocity MSE | Uniform forward Euler integration |

All three restoration CNNs have **14,986,435 trainable parameters** and use **30,000 updates**, with seeds 0 and 1. Initialization, image draws, mask-pair draws, and horizontal flips are matched within a seed. Matching update counts does not match training FLOPs: perceptual regression additionally evaluates a frozen feature network.

The paired masks each remove 1,638 of 4,096 pixels. They preserve missing area, quadrant counts, and the central mask pattern, with annotation-based matching tolerances. After matching, the protocol retains **2,812 training**, **124 development**, and **3,585 official-test** images. Full conditions and configuration selection are in [Experimental protocol](docs/PROTOCOL.md).

## Archived test results

The FM8 setting was selected on development LPIPS. Results below average two training seeds and weight the two geometries equally.

| Method | NFE | LPIPS ↓ | Masked PSNR ↑ | Batch-1 median latency |
|---|---:|---:|---:|---:|
| Pixel regression | 1 | 0.13119 | 15.906 dB | 3.77 ms |
| Perceptual regression | 1 | 0.12046 | 15.723 dB | 3.82 ms |
| Conditional FM | 8 | **0.10621** | **16.159 dB** | 28.41 ms |

FM8 reduces mean LPIPS by **19.0%** relative to pixel regression and **11.8%** relative to the tested perceptual regressor. The associated PSNR gains are **0.254 dB** and **0.437 dB**, respectively, calculated from unrounded results. Its single-image inference latency is approximately **7.4×** that of perceptual regression on the recorded RTX 4090 system.

These are overall averages. Against pixel regression, the PSNR gain is concentrated in fragmented masks. The perceptual-regression control was designed after the original FM/pixel-regression test results had been observed. Its coefficient was selected using development data before its own test evaluation. The results do not establish universal superiority, optimal budgets outside the tested settings, distributional realism, or semantic diversity. Image-bootstrap intervals condition on the two trained seeds.

Sources: [test summary](results/test_summary.json), [timing records](results/latency.json), and [perceptual-loss selection](results/perceptual_selection.json).

## Training diagnostics

![Training objectives and reconstruction metrics](figures/training_dynamics.png)

*Training-probe objectives decrease while development-probe objectives rise. Task metrics follow a different trajectory. Task-quality curves use one fixed mask pair per image; the final development NFE sweep uses four. The final comparisons retain the predetermined 30k endpoint.*

## Repository layout

```text
GeoStep/             Original model, mask, training, evaluation, and analysis scripts
                     plus historical protocol files; see docs/CODE_MAP.md
results/             Archived test summaries, selection/timing records, and plot CSVs
data/                Frozen split manifest; no original photographs or annotations
figures/             Numerical figures rebuilt from the included tables
scripts/             Lightweight release checks, result display, and plotting
tests/               Synthetic CPU checks for masks, architecture, and sampling
provenance/          SHA-256 records for copied source and data files
docs/                Protocol, execution prerequisites, and code map
```

`GeoStep` is the internal experiment name. Early-stage helpers remain where required by the formal scripts' imports. Their presence does not make pilot settings part of the final paper comparison. Remote collection scripts, server launchers, unrelated follow-up experiments, checkpoints, and dataset images are excluded.

## Environment and validation

The recorded training environment used **PyTorch 2.7.1+cu128**, CUDA, BF16, and an RTX 4090. [requirements.txt](requirements.txt) lists direct dependencies rather than a complete environment lock. Install a compatible PyTorch/TorchVision CUDA build for the target machine.

For synthetic CPU checks after installing the model dependencies:

```bash
python -m unittest discover -s tests -v
```

These checks validate implementation properties; they do not reproduce model quality or validate the original CUDA/BF16 execution. Release checks and their limitations are documented in [Validation](docs/VALIDATION.md).

## Data and attribution

Obtain the dataset from the [Oxford-IIIT Pet dataset page](https://www.robots.ox.ac.uk/~vgg/data/pets/). The repository includes only the frozen image-ID split manifest, not the dataset itself. See [Data instructions](data/README.md).

The study builds on [Flow Matching](https://arxiv.org/abs/2210.02747), [U-Net](https://arxiv.org/abs/1505.04597), [perceptual feature losses](https://arxiv.org/abs/1603.08155), and [LPIPS](https://arxiv.org/abs/1801.03924). Dependency and dataset terms remain with their respective projects. A project-wide license has not yet been specified for this release.

Citation metadata for this code repository is provided in [CITATION.cff](CITATION.cff).
