#!/usr/bin/env python3
"""Display archived aggregates; no training, inference, or metric evaluation."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = json.loads((ROOT / 'results/test_summary.json').read_text())
    timings = json.loads((ROOT / 'results/latency.json').read_text())['rows']
    print('Archived official-test results: two-seed means, 3,585 retained images.')
    print('Perceptual regression is a subsequent control; see docs/PROTOCOL.md.')
    print(f'{"Geometry":<12} {"Model":<24} {"LPIPS down":>12} {"PSNR dB up":>12}')
    for row in rows:
        print(f'{row["geometry"]:<12} {row["model"]:<24} '
              f'{row["lpips_truth"]:12.5f} {row["sample_psnr"]:12.3f}')
    print('\nRecorded GPU latency, one output per image:')
    for row in timings:
        batch = row['batch_size']
        print(f'{row["model"]:<24} batch={batch:2} '
              f'median={row["batch_median_ms"]:.3f} ms/batch '
              f'({row["batch_median_ms"] / batch:.3f} ms/image)')


if __name__ == '__main__':
    main()
