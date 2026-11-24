"""Audit the portable report/evidence handoff without a GPU or raw dataset.

This verifies saved records and packaged bytes, not float prediction rescoring.
Run from any directory: uv run /path/to/repo/scripts/verify_handoff.py
"""
from pathlib import Path
import collections
import csv
import hashlib
import json
import math
import re
import statistics
import sys
import zipfile
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / 'results/rahil/frozen-test'
HELPER = ROOT / 'tools/frozen-test'
REPORT = ROOT / 'reports/intermediate-2026-11-24'
METHODS = ['C0', 'C1', 'C2', 'C3', 'M0', 'M1', 'M2', 'M3']
METRICS = ['psnr_mu', 'psnr_l', 'ssim_mu', 'ssim_l', 'mae_mu']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(line for line in handle if not line.startswith('#')))


def main():
    artifacts = read_json(ROOT / 'results/rahil/artifact_sha256.json')
    for name, digest in artifacts.items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and path.is_file(), f'Missing/unsafe artifact: {name}')
        require(sha(path) == digest, f'Changed frozen handoff artifact: {name}')
    print(f'PASS: {len(artifacts)} report/helper/evidence artifact hashes')

    code_hashes = read_json(HELPER / 'helper_sha256.json')
    for name, digest in code_hashes.items():
        require(sha(HELPER / name) == digest, f'Frozen helper changed: {name}')
    protocol = read_json(TEST / 'test_protocol_frozen.json')
    require((TEST / 'test_protocol_frozen.json').read_bytes() == (HELPER / 'test_protocol_frozen.json').read_bytes(), 'Protocol copies differ')
    require(sha(HELPER / 'training_split_frozen.json') == protocol['training_split_sha256'], 'Frozen split hash differs')
    require((ROOT / 'configs/split.json').read_bytes() == (HELPER / 'training_split_frozen.json').read_bytes(), 'Training/validation split differs')
    for model, entry in protocol['checkpoints'].items():
        require(sha(ROOT / 'release/checkpoints' / entry['path']) == entry['sha256'], f'Released {model} checkpoint differs from frozen hash')
    print('PASS: preserved helper/protocol/split and four released checkpoint byte hashes')

    runtime = read_json(TEST / 'runtime.json')
    manifest = read_json(TEST / 'input_manifest.json')
    status = read_json(TEST / 'status.json')
    completed = read_json(TEST / 'completed_scenes.json')
    require(runtime['protocol_sha256'] == manifest['protocol_sha256'] == sha(TEST / 'test_protocol_frozen.json'), 'Runtime/protocol hashes differ')
    require(runtime['input_manifest_sha256'] == sha(TEST / 'input_manifest.json'), 'Input-manifest hash differs')
    require(manifest['illustration_crops_sha256'] == sha(HELPER / 'input_illustration_crops.json'), 'Input-selected crops differ')
    require(manifest['status'] == 'ready' and not manifest['exact_input_overlaps'] and not manifest['similarity_flags'], 'Unresolved overlap status')
    require(manifest['all_training_ids_checked'] == [f'{i:03d}' for i in range(1, 75)], 'Training overlap audit is incomplete')
    keys = protocol['test_scene_keys']
    require(len(keys) == 15 and len(set(keys)) == 15, 'Test list is incomplete/duplicated')
    require([x['scene_key'] for x in manifest['scenes']] == keys == [x['scene'] for x in completed], 'Scene manifests differ')
    require(status['status'] == 'complete' and status['method_scene_pairs'] == 120 and status['completed_scenes'] == 15, 'Run incomplete')
    for entry, record in zip(manifest['scenes'], completed):
        require(entry['crop']['xywh'] == record['crop_xywh'], 'Crop moved after preparation')
        require(set(record['prediction_sha256']) == set(METHODS), 'Prediction hash grid is incomplete')
        require(all(re.fullmatch('[0-9a-f]{64}', x) for x in record['prediction_sha256'].values()), 'Malformed prediction hash')
        for name in ['comparison.png', 'errors.png']:
            require((TEST / 'figures' / entry['scene_key'] / name).is_file(), f'Missing figure for {entry["scene_key"]}')

    scores = rows(TEST / 'per_scene_metrics.csv')
    summary = rows(TEST / 'summary_metrics.csv')
    index = {(r['scene'], r['method']): r for r in scores}
    require(len(scores) == len(index) == 120 and set(index) == {(s, m) for s in keys for m in METHODS}, 'Score grid incomplete or duplicated')
    require(len(summary) == 24 and len({(r['group'], r['method']) for r in summary}) == 24, 'Aggregate grid incomplete or duplicated')
    require({(r['group'], r['method']) for r in summary} == {(g, m) for g in ['ALL15', 'PAPER', 'EXTRA'] for m in METHODS}, 'Unexpected aggregate groups')
    for row in scores:
        require(all(math.isfinite(float(row[m])) for m in METRICS), 'Nonfinite metric')
        require(0 <= float(row['ssim_mu']) <= 1 and 0 <= float(row['ssim_l']) <= 1 and float(row['mae_mu']) >= 0, 'Invalid metric range')
    maxdiff = 0.0
    for row in summary:
        selected = [r for r in scores if r['method'] == row['method'] and (row['group'] == 'ALL15' or r['scene'].startswith(row['group'] + '/'))]
        require(len(selected) == int(row['n_scenes']), 'Aggregate count differs')
        for metric in METRICS:
            delta = abs(statistics.mean(float(r[metric]) for r in selected) - float(row[metric]))
            maxdiff = max(maxdiff, delta)
            require(delta < 1e-12, f'Aggregate differs: {row["group"]}/{row["method"]}/{metric}')
    differences = [float(index[s, 'M3']['psnr_mu']) - float(index[s, 'M2']['psnr_mu']) for s in keys]
    require(sum(x > 0 for x in differences) == 9 and sum(x < 0 for x in differences) == 6, 'Win/loss count differs')
    require(abs(statistics.mean(differences) - 0.08561734341304496) < 1e-12, 'Reported paired mean differs')
    reported_pairs = rows(TEST / 'paired_M3_minus_M2_audited.csv')
    require([r['scene'] for r in reported_pairs] == keys, 'Paired CSV scenes differ')
    for scene, delta, row in zip(keys, differences, reported_pairs):
        require(abs(float(row['delta_psnr_mu']) - delta) < 1e-12, f'Paired delta differs: {scene}')
    print(f'PASS: 120 unique scores; all 24 aggregates recomputed (max error {maxdiff:.2e}); 9 wins/6 losses; M3-M2 {statistics.mean(differences):+.7f} dB')

    duplicates = collections.defaultdict(list)
    for record in completed:
        duplicates[record['target_sha256']].append(record['scene'])
    actual_pairs = {frozenset(v) for v in duplicates.values() if len(v) > 1}
    expected_pairs = {frozenset(['EXTRA/003', 'PAPER/LadySitting']), frozenset(['EXTRA/005', 'PAPER/PeopleStanding'])}
    require(actual_pairs == expected_pairs, 'Duplicate-target disclosure differs')

    regional = ROOT / 'results/rahil/validation-regions'
    saved = {(r['scene'], r['model']): r for r in rows(ROOT / 'results/tables/variants_val.csv') if r['scene'] != 'mean'}
    audited = rows(regional / 'full_image_score_audit.csv')
    require(len(audited) == 20 and {(r['scene_id'], r['model']) for r in audited} == set(saved), 'Validation audit grid differs')
    for row in audited:
        expected = saved[row['scene_id'], row['model']]
        require(row['within_tolerance'] == 'True', 'Validation audit recorded a failure')
        for metric in ['psnr_mu', 'ssim_mu']:
            require(abs(float(row[metric + '_recomputed']) - float(expected[metric])) < 1e-12, f'Validation record differs: {metric}')
    config = read_json(regional / 'config.json')
    for key, name in [('analysis', 'analysis_code.py'), ('baseline', 'classical_baselines.py'), ('fallback', 'motion_rejection.py')]:
        require(sha(regional / name) == config['code_sha256'][key], 'Regional analysis code differs')
    require(sha(regional / 'saved_validation_metrics.csv') == config['expected_metrics_sha256'], 'Regional expected metrics differ')
    print('PASS: 20 saved validation score checks, regional code hashes and disclosed test duplicate pairs')

    source = (REPORT / 'main.tex').read_text()
    for name in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', source):
        require((REPORT / name).is_file(), f'Missing LaTeX image: {name}')
    with zipfile.ZipFile(REPORT / 'Overleaf_update_files.zip') as archive:
        require(archive.testzip() is None, 'Overleaf ZIP CRC failed')
        require(len(archive.namelist()) == 10, 'Unexpected Overleaf payload')
        for name in archive.namelist():
            require(archive.read(name) == (REPORT / name).read_bytes(), f'Overleaf ZIP source differs: {name}')
    docs = [ROOT / p for p in ['README.md', 'AGENTS.md', 'docs/HANDOFF.md', 'docs/REPRODUCING.md', 'results/TEST_RESULTS.md', 'results/rahil/README.md', 'reports/intermediate-2026-11-24/README.md', 'tools/frozen-test/README.md', 'results/by_step/README.md', 'results/by_step/R1-R3_data_and_split/README.md', 'results/by_step/R4_classical_C0-C3/README.md']]
    links = 0
    for path in docs:
        content = path.read_text()
        require('\u2014' not in content, f'Em dash in new/edited guide: {path.relative_to(ROOT)}')
        for target in re.findall(r'\[[^\]]*\]\(([^\s)]+)\)', content):
            if re.match(r'[a-zA-Z][a-zA-Z0-9+.-]*:', target) or target.startswith('#'):
                continue
            target = unquote(target.split('#')[0])
            require((path.parent / target).exists(), f'Broken link in {path.relative_to(ROOT)}: {target}')
            links += 1
    print(f'PASS: three report figures, 10-file Overleaf ZIP and {links} local documentation links')
    print('LIMIT: saved-record audit only; full test float arrays were not transferred or rescored. Human writing/contribution/disclosure review and submission confirmation remain.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, csv.Error, zipfile.BadZipFile) as exc:
        print(f'FAIL: {exc}', file=sys.stderr)
        raise SystemExit(1)
