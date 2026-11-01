# Release validation

Validation date: 2026-11-01 (repository framework and final-report presentation update).

The public packaging checks were run locally with Python 3.13.5, PyTorch 2.9.1, and TorchVision 0.24.1 on CPU. This differs from the recorded Linux CUDA/BF16 training environment and is not a claim of bitwise reproduction.

## Completed checks

- SHA-256 verification of 73 original scientific files and four added paper/display assets. The report and supplementary PDFs match the submitted files byte for byte.
- Python and JSON parsing, local documentation links, split-manifest identity, and aggregate-result arithmetic.
- Four synthetic CPU tests: matched-mask invariants, 14,986,435-parameter backbone/output shape, Euler evaluation counts with exact known-pixel preservation, and mean-of-sample PSNR behavior.
- All nine manuscript plots regenerated from included measurements and display tiles. Their PNG exports match the final manuscript assets byte for byte in this local environment.
- The three plotting programs run in isolated processes to prevent Matplotlib style settings from leaking across figures.
- README rendering inspected locally, including the result table and qualitative examples.
- The approved repository framework was checked in HTML and PNG, with editable SVG text. The standalone generator reproduces the approved SVG; submitted PDF hashes remain unchanged.
- Standard-library release checks passed on the updated working tree. The 2026-10-30 release was also checked from a clean Git checkout.

The synthetic metric test substitutes a small distance function for the perceptual network. It tests PSNR aggregation and does not claim to validate LPIPS weights or perceptual quality. CPU sampling tests disable the CUDA autocast context only inside the test; archived research source is unchanged.

The GitHub Actions workflow checks archive integrity and regenerates figures on Python 3.11. The current online status is available in the repository Actions tab. Local plotting used NumPy 2.4.1 and Matplotlib 3.10.0.

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
