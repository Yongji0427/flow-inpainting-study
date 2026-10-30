# Release validation

Validation date: 2026-10-30.

The public packaging checks were run locally with Python 3.13.5, PyTorch 2.9.1, and TorchVision 0.24.1 on CPU. This differs from the recorded Linux CUDA/BF16 training environment and is not a claim of bitwise reproduction.

## Completed checks

- SHA-256 verification of 73 copied source, protocol, manifest, and result files.
- Python and JSON parsing, local documentation links, split-manifest identity, and aggregate-result arithmetic.
- Four synthetic CPU tests: matched-mask invariants, 14,986,435-parameter backbone/output shape, Euler evaluation counts with exact known-pixel preservation, and mean-of-sample PSNR behavior.
- Numerical figures regenerated from included CSV tables and visually inspected.
- Standard-library release checks repeated from a clean Git checkout.

The synthetic metric test substitutes a small distance function for the perceptual network. It tests PSNR aggregation and does not claim to validate LPIPS weights or perceptual quality. CPU sampling tests disable the CUDA autocast context only inside the test; archived research source is unchanged.

## Not run for this release

No training, new image inference, full-test metric recomputation, GPU timing, CUDA/BF16 equivalence check, or complete reconstruction of historical training artifacts was performed. Historical verification JSON files describe the original experiments, not new checks performed during publication.

## Local checks

```bash
python scripts/check_release.py
python scripts/show_results.py
python -m unittest discover -s tests -v
python scripts/plot_results.py
```

The first two commands require only the Python standard library. The last two require the dependencies described in the README.
