#!/usr/bin/env python3
"""Read-only standard-library checks for the published source/result package."""
import ast
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads((ROOT / path).read_text())


def main():
    manifest = read_json('provenance/source_manifest.json')
    paper_assets = read_json('provenance/paper_assets.json')
    for row in manifest['files'] + paper_assets['files']:
        path = ROOT / row['path']
        assert path.is_file(), f'Missing archived file: {row["path"]}'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], row['path']
    sources = list(ROOT.rglob('*.py'))
    local_modules = {p.stem for p in (ROOT / 'GeoStep').glob('*.py')}
    for path in sources:
        if '.git' in path.parts or '__pycache__' in path.parts:
            continue
        tree = ast.parse(path.read_text(), filename=str(path.relative_to(ROOT)))
        if path.parent == ROOT / 'GeoStep':
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    module = node.module.split('.')[0]
                    if module in local_modules:
                        assert (ROOT / 'GeoStep' / (module + '.py')).is_file()
    for path in ROOT.rglob('*.json'):
        if '.git' not in path.parts:
            json.loads(path.read_text())
    with (ROOT / 'data/splits.csv').open(newline='') as handle:
        records = list(csv.DictReader(handle))
    assert len({r['image_id'] for r in records}) == len(records)
    assert Counter(r['split'] for r in records) == {'train': 2924, 'val': 730, 'test': 3669}
    split_sha = hashlib.sha256((ROOT / 'data/splits.csv').read_bytes()).hexdigest()
    assert split_sha == read_json('GeoStep/official_test_protocol_v9.json')['source_manifest_expected_sha256']
    rows = read_json('results/test_summary.json')
    assert len(rows) == 9
    for model in ['MSE Regression', 'Perceptual Regression', 'FM8']:
        by_geometry = {r['geometry']: r for r in rows if r['model'] == model}
        assert set(by_geometry) == {'large', 'fragmented', 'overall'}
        for metric in ['lpips_truth', 'sample_psnr']:
            expected = (by_geometry['large'][metric] + by_geometry['fragmented'][metric]) / 2
            assert math.isclose(expected, by_geometry['overall'][metric], abs_tol=1e-12)
    with (ROOT / 'results/development/figures_nfe_source.csv').open(newline='') as handle:
        sweep = list(csv.DictReader(handle))
    for seed in ['0', '1']:
        for geometry in ['large', 'fragmented']:
            group = [r for r in sweep if r['model'] == 'fm' and r['seed'] == seed and r['geometry'] == geometry]
            assert {int(r['nfe']) for r in group} == {1, 2, 4, 8, 16, 32}
            assert int(min(group, key=lambda r: float(r['lpips']))['nfe']) == 8
            assert int(max(group, key=lambda r: float(r['psnr_db']))['nfe']) == 2
    for path in ROOT.rglob('*.md'):
        content = path.read_text()
        links = re.findall(r'\]\(([^)]+)\)', content)
        links += re.findall(r'(?:href|src)=[\"\']([^\"\']+)[\"\']', content)
        for link in links:
            if '://' not in link and not link.startswith('#'):
                target = link.split('#', 1)[0]
                assert (path.parent / target).exists(), f'Broken link: {path.relative_to(ROOT)} -> {link}'
    print(f'PASS: {len(manifest["files"])} research hashes and {len(paper_assets["files"])} paper/display hashes,')
    print('Python/JSON syntax, split identity,')
    print('aggregate arithmetic, four FM NFE profiles, and local documentation links.')
    print('No models were trained or evaluated. Historical runtime prerequisites are not verified here.')


if __name__ == '__main__':
    main()
