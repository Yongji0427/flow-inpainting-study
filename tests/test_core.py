"""Synthetic CPU checks, not training or a reproduction of reported quality."""
import contextlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'GeoStep'))
import pilot
from capacity15_models import LargeUNet
from matched_masks import candidate_parameters, make_masks
from followup import scores

torch.set_num_threads(2)


class ConstantField(nn.Module):
    def __init__(self):
        super().__init__()
        self.times = []

    def forward(self, z, y, m, t):
        self.times.append(float(t[0]))
        return torch.full_like(z, .125)


class DummyPerceptual(nn.Module):
    def forward(self, a, b):
        return (a - b).square().mean((1, 2, 3), keepdim=True)


class CoreTests(unittest.TestCase):
    def test_matched_mask_invariants(self):
        config = json.loads((ROOT / 'GeoStep/matched_mask_protocol.json').read_text())
        candidates = candidate_parameters(config)
        for index in np.linspace(0, len(candidates) - 1, 12, dtype=int):
            big, fragmented = make_masks(candidates[index])
            self.assertEqual(int(big.sum()), 1638)
            self.assertEqual(int(fragmented.sum()), 1638)
            np.testing.assert_array_equal(big[24:40, 24:40], fragmented[24:40, 24:40])
            for y in [0, 32]:
                for x in [0, 32]:
                    self.assertEqual(int(big[y:y+32, x:x+32].sum()), int(fragmented[y:y+32, x:x+32].sum()))

    def test_paper_backbone_shape_and_parameter_count(self):
        torch.manual_seed(0)
        model = LargeUNet().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 14986435)
        x = torch.zeros(1, 3, 64, 64)
        with torch.no_grad():
            result = model(x, x, torch.zeros(1, 1, 64, 64), torch.zeros(1))
        self.assertEqual(tuple(result.shape), (1, 3, 64, 64))
        self.assertTrue(torch.isfinite(result).all())

    def test_euler_budget_and_known_pixel_preservation(self):
        x = torch.linspace(-1, 1, 3 * 64 * 64).reshape(1, 3, 64, 64)
        m = torch.zeros(1, 1, 64, 64)
        m[:, :, 8:40, 12:44] = 1
        eps = torch.zeros_like(x)
        for nfe in [1, 2, 8]:
            model = ConstantField()
            with patch.object(pilot, 'autocast', contextlib.nullcontext):
                result = pilot.predict(model, 'fm', x, m, eps, nfe)
            self.assertEqual(len(model.times), nfe)
            np.testing.assert_allclose(model.times, np.arange(nfe) / nfe)
            torch.testing.assert_close(result, x * (1 - m) + .125 * m, rtol=0, atol=0)

    def test_psnr_is_mean_of_sample_scores(self):
        x = torch.zeros(1, 3, 16, 16)
        m = torch.ones(1, 1, 16, 16)
        offsets = torch.tensor([.2, .4, .6, .8])[:, None, None, None, None]
        pred = x[None] + offsets
        result = scores(pred, pred, x, m, DummyPerceptual())
        expected = (-10 * torch.log10((offsets.flatten() / 2).square())).mean().item()
        self.assertAlmostEqual(float(result['sample_psnr'][0]), expected, places=5)
        score_of_mean = -10 * np.log10((float(offsets.mean()) / 2) ** 2)
        self.assertGreater(abs(expected - score_of_mean), .1)


if __name__ == '__main__':
    unittest.main()
