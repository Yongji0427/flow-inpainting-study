#!/usr/bin/env python3
"""Redraw numerical figures from archived CSV values, without model evaluation."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/development'
OUTPUT = ROOT / 'figures'
COLORS = {'fm': '#287AA4', 'regression': '#BD7137', 'perceptual_regression': '#8865A3'}
NFE = [1, 2, 4, 8, 16, 32]
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                     'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': .7, 'pdf.fonttype': 42, 'legend.frameon': False,
                     'savefig.facecolor': 'white'})


def read(name):
    with (DATA / name).open(newline='') as handle:
        return list(csv.DictReader(handle))


def save(fig, name):
    OUTPUT.mkdir(exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(OUTPUT / f'{name}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def nfe_figure():
    rows = read('figures_nfe_source.csv')
    fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharex=True)
    for col, geometry in enumerate(['large', 'fragmented']):
        for row, (metric, label) in enumerate([('lpips', 'LPIPS ↓'), ('psnr_db', 'Masked PSNR (dB) ↑')]):
            ax = axes[row, col]
            curves = []
            for seed in ['0', '1']:
                group = {int(r['nfe']): r for r in rows if r['model'] == 'fm' and r['seed'] == seed and r['geometry'] == geometry}
                values = [float(group[n][metric]) for n in NFE]
                curves.append(values)
                ax.plot(NFE, values, color=COLORS['fm'], alpha=.3, lw=.8)
            ax.plot(NFE, np.mean(curves, axis=0), 'o-', color=COLORS['fm'], lw=1.6, ms=4, label='FM')
            for model, title, style in [('regression', 'Pixel regression', '--'), ('perceptual_regression', 'Perceptual regression', ':')]:
                values = [float(r[metric]) for r in rows if r['model'] == model and r['geometry'] == geometry]
                assert len(values) == 2
                ax.axhline(np.mean(values), color=COLORS[model], linestyle=style, lw=1.2, label=title)
            ax.set_xscale('log', base=2)
            ax.set_xticks(NFE, [str(n) for n in NFE])
            ax.set_ylabel(label)
            ax.grid(axis='y', alpha=.2)
            if row == 0:
                ax.set_title('Large hole' if geometry == 'large' else 'Fragmented mask', loc='left')
            else:
                ax.set_xlabel('FM network evaluations (NFE)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .94))
    save(fig, 'nfe_quality')


def training_figure():
    objectives = read('figures_training_objectives.csv')
    tasks = read('figures_task_learning.csv')
    steps = [10000, 20000, 30000]
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.2))
    for col, model in enumerate(['fm', 'regression']):
        ax = axes[0, col]
        for split, label, style in [('train_probe', 'Train probe', '-'), ('dev', 'Development', '--')]:
            curves = []
            for seed in ['0', '1']:
                group = {int(r['step']): r for r in objectives if r['model'] == model and r['seed'] == seed and r['split'] == split and r['variant'] == 'ema'}
                values = [float(group[k]['loss']) for k in steps]
                curves.append(values)
                ax.plot([10, 20, 30], values, style, color=COLORS[model], alpha=.3, lw=.8)
            ax.plot([10, 20, 30], np.mean(curves, axis=0), style, marker='o', color=COLORS[model], lw=1.6, ms=4, label=label)
        ax.set_title('FM velocity objective' if model == 'fm' else 'Pixel-regression objective', loc='left')
        ax.set_ylabel('Velocity MSE' if model == 'fm' else 'Image MSE')
        ax.legend(fontsize=8)
    for col, (metric, title, label) in enumerate([('lpips', 'Development perceptual similarity', 'LPIPS ↓'), ('psnr_db', 'Development pixel fidelity', 'Masked PSNR (dB) ↑')]):
        ax = axes[1, col]
        for model in ['fm', 'regression']:
            curves = []
            for seed in ['0', '1']:
                group = {int(r['step']): r for r in tasks if r['model'] == model and r['seed'] == seed}
                values = [float(group[k][metric]) for k in steps]
                curves.append(values)
                ax.plot([10, 20, 30], values, color=COLORS[model], alpha=.3, lw=.8)
            ax.plot([10, 20, 30], np.mean(curves, axis=0), 'o-', color=COLORS[model], lw=1.6, ms=4, label='FM8' if model == 'fm' else 'Pixel regression')
        ax.set_title(title, loc='left')
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([10, 20, 30])
        ax.set_xlabel('Training updates (thousands)')
        ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    save(fig, 'training_dynamics')


if __name__ == '__main__':
    nfe_figure()
    training_figure()
    print('Rebuilt two figures from archived CSV tables; no inference or new metrics.')
