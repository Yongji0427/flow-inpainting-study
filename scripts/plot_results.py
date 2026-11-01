#!/usr/bin/env python3
"""Rebuild all nine manuscript plots from archived inputs; no model evaluation."""
from pathlib import Path
import subprocess
import sys

SCRIPTS = Path(__file__).resolve().parent

if __name__ == '__main__':
    for script in ['plot_quantitative_figures.py', 'plot_sampling_budget.py',
                   'plot_qualitative_results.py']:
        subprocess.run([sys.executable, str(SCRIPTS / script)], check=True)
    print('Rebuilt Figures 1b, 3-5 and S1-S5 in PDF, SVG and PNG.')
    print('Figure 1a is preserved artwork; Figure 2 has a separate HTML/SVG generator. No training or inference.')
