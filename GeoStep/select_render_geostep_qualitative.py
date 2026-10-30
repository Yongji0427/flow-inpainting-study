"""Select and render an auditable median GeoStep official-test example.

Selection uses only already-saved per-image metrics. It ranks every retained test
image by (1) mean FM8 LPIPS and (2) mean LPIPS improvement over MSE regression,
after averaging both seeds, both mask geometries, and all four mask pairs.  The
chosen image minimizes the L1 distance of the two percentile ranks to 0.5.

The output prediction grid reruns only the chosen image with frozen seed-0 EMA
checkpoints, mask pair 0, and the first fixed FM noise sample.  It does not train,
tune, select a checkpoint, or alter any existing official-test artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from capacity15_models import LargeUNet
from followup import noise_bank
from pilot import predict


ROOT = Path("/data/GeoStep")
V9 = ROOT / "runs" / "official_test_two_seeds_v9"
PER = ROOT / "runs" / "perceptual_regression_30k_v10"
OUT = ROOT / "runs" / "qualitative_median_v11"


def read(path: Path):
    return json.loads(path.read_text())


def per_image(rows: list[dict], metric: str) -> np.ndarray:
    groups: dict[int, list[float]] = {}
    for row in rows:
        groups.setdefault(row["image_index"], []).append(row[metric])
    return np.asarray([np.mean(groups[i]) for i in sorted(groups)], dtype=np.float64)


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / max(len(values) - 1, 1)


def select() -> tuple[int, dict]:
    fm, reg = [], []
    for seed in (0, 1):
        fm_rows = read(V9 / f"fm_seed{seed}" / "fm_step30000_final_rows.json")
        reg_rows = read(V9 / f"regression_seed{seed}" / "regression_step30000_final_rows.json")
        fm.append(per_image(fm_rows, "lpips_truth"))
        reg.append(per_image(reg_rows, "lpips_truth"))
    fm_mean = np.mean(fm, axis=0)
    reg_mean = np.mean(reg, axis=0)
    gain = reg_mean - fm_mean
    q_quality = percentile_ranks(fm_mean)
    q_gain = percentile_ranks(gain)
    centrality = np.abs(q_quality - 0.5) + np.abs(q_gain - 0.5)
    index = int(np.argmin(centrality))
    metadata = read(V9 / "image_mask_metadata.json")[index]
    result = {
        "selection_rule": (
            "minimum L1 distance to the 50th percentile in both mean FM8 LPIPS "
            "and mean LPIPS improvement over MSE, after averaging two seeds, "
            "two geometries, and four mask pairs"
        ),
        "image_index": index,
        "image_id": metadata["image_id"],
        "fm_lpips_mean": float(fm_mean[index]),
        "mse_lpips_mean": float(reg_mean[index]),
        "lpips_gain_mean": float(gain[index]),
        "fm_quality_percentile": float(q_quality[index]),
        "gain_percentile": float(q_gain[index]),
        "centrality": float(centrality[index]),
        "selection_uses_model_outputs": True,
        "selection_purpose": "descriptive median qualitative example only",
        "display": {
            "seed": 0,
            "mask_pair": 0,
            "fm_noise_sample": 0,
            "nfe": 8,
        },
    }
    return index, result


def load_model(path: Path) -> LargeUNet:
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    model = LargeUNet().cuda().eval()
    model.load_state_dict(checkpoint["ema"])
    return model


def to_image(tensor: torch.Tensor, scale: int = 4) -> Image.Image:
    array = ((tensor.detach().cpu().clamp(-1, 1).permute(1, 2, 0).numpy() + 1) * 127.5).round().astype(np.uint8)
    return Image.fromarray(array).resize((64 * scale, 64 * scale), Image.Resampling.NEAREST)


def get_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


@torch.no_grad()
def render(index: int, selection: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    cache = torch.load(V9 / "prepared_test.pt", weights_only=True)
    x = cache["dev"][index : index + 1].float().cuda() / 127.5 - 1
    bank = cache["masks"]
    noise = noise_bank((len(cache["dev"]), 4, 3, 64, 64), 202610225)

    models = {
        "MSE Regression": load_model(ROOT / "runs/regression15_30k_v7/seed0/regression_step30000.pt"),
        "Perceptual Regression": load_model(PER / "alpha100/seed0/regression_step30000.pt"),
        "FM8": load_model(ROOT / "runs/capacity15_30k_v6/seed0/fm_step30000.pt"),
    }

    columns = ["Ground truth", "Masked input", "MSE Regression", "Perceptual Regression", "FM8"]
    geometries = [("Large hole", 0), ("Fragmented", 1)]
    tiles: dict[tuple[str, str], Image.Image] = {}
    for geometry, gi in geometries:
        mask = bank[index, 0, gi][None, None].float().cuda()
        visible = x * (1 - mask)
        eps = noise[0, index, 0][None].cuda()
        pred_mse = predict(models["MSE Regression"], "regression", x, mask, torch.zeros_like(x), 1)[0]
        pred_per = predict(models["Perceptual Regression"], "regression", x, mask, torch.zeros_like(x), 1)[0]
        pred_fm = predict(models["FM8"], "fm", x, mask, eps, 8)[0]
        values = [x[0], visible[0], pred_mse, pred_per, pred_fm]
        for label, value in zip(columns, values):
            tiles[geometry, label] = to_image(value)

    cell, gap, left, top = 256, 18, 150, 82
    width = left + len(columns) * cell + (len(columns) - 1) * gap + 28
    height = top + 2 * cell + gap + 88
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = get_font(26, True)
    label_font = get_font(20, True)
    note_font = get_font(15, False)
    for j, label in enumerate(columns):
        x0 = left + j * (cell + gap)
        box = draw.textbbox((0, 0), label, font=label_font)
        draw.text((x0 + (cell - (box[2] - box[0])) / 2, 30), label, fill="#17233d", font=label_font)
    for i, (geometry, _) in enumerate(geometries):
        y0 = top + i * (cell + gap)
        draw.text((20, y0 + cell / 2 - 14), geometry, fill="#17233d", font=label_font)
        for j, label in enumerate(columns):
            x0 = left + j * (cell + gap)
            canvas.paste(tiles[geometry, label], (x0, y0))
            draw.rectangle((x0, y0, x0 + cell - 1, y0 + cell - 1), outline="#8a98aa", width=2)
    title = f"Median official-test example: {selection['image_id']}"
    draw.text((20, 4), title, fill="#0b315d", font=title_font)
    note = (
        "Selected by two-seed / two-geometry / four-pair aggregate ranks; "
        "shown: seed 0, mask pair 0, FM first fixed noise sample."
    )
    draw.text((20, height - 48), note, fill="#4b5b70", font=note_font)
    canvas.save(OUT / "qualitative_median.png", quality=95)
    (OUT / "selection.json").write_text(json.dumps(selection, indent=2))


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite {OUT}")
    index, selection = select()
    print(json.dumps(selection, indent=2), flush=True)
    render(index, selection)


if __name__ == "__main__":
    main()
