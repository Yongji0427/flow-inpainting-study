"""Compact per-image win-rate analysis from frozen GeoStep test rows."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path("/data/GeoStep")
V9 = ROOT / "runs" / "official_test_two_seeds_v9"
PER = ROOT / "runs" / "perceptual_regression_30k_v10/post_test_evaluation"
OUT = ROOT / "runs" / "per_image_win_rates_v11.json"


def read(path: Path):
    return json.loads(path.read_text())


def values(rows: list[dict], geometry: str, metric: str) -> np.ndarray:
    allowed = {"large", "fragmented"} if geometry == "overall" else {geometry}
    grouped: dict[int, list[float]] = {}
    for row in rows:
        if row["geometry"] in allowed:
            grouped.setdefault(row["image_index"], []).append(row[metric])
    return np.asarray([np.mean(grouped[i]) for i in sorted(grouped)], dtype=np.float64)


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [center - half, center + half]


def main() -> None:
    sources = {}
    for seed in (0, 1):
        sources[seed, "MSE Regression"] = read(V9 / f"regression_seed{seed}/regression_step30000_final_rows.json")
        sources[seed, "Perceptual Regression"] = read(PER / f"seed{seed}/regression_step30000_final_rows.json")
        sources[seed, "FM8"] = read(V9 / f"fm_seed{seed}/fm_step30000_final_rows.json")

    output = []
    for baseline in ("MSE Regression", "Perceptual Regression"):
        for geometry in ("large", "fragmented", "overall"):
            agg = {}
            for model in (baseline, "FM8"):
                lpips = [values(sources[seed, model], geometry, "lpips_truth") for seed in (0, 1)]
                psnr = [values(sources[seed, model], geometry, "sample_psnr") for seed in (0, 1)]
                agg[model] = (np.mean(lpips, axis=0), np.mean(psnr, axis=0))
            base_lpips, base_psnr = agg[baseline]
            fm_lpips, fm_psnr = agg["FM8"]
            wins_lpips = fm_lpips < base_lpips
            wins_psnr = fm_psnr > base_psnr
            wins_joint = wins_lpips & wins_psnr
            n = len(wins_joint)
            row = {
                "baseline": baseline,
                "candidate": "FM8",
                "geometry": geometry,
                "images": n,
            }
            for name, mask in (
                ("lpips_win", wins_lpips),
                ("psnr_win", wins_psnr),
                ("joint_win", wins_joint),
            ):
                count = int(mask.sum())
                row[name + "_count"] = count
                row[name + "_rate"] = count / n
                row[name + "_wilson95"] = wilson(count, n)
            output.append(row)
    OUT.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
