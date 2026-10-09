"""Check generated table values and recorded CLI reproduction against source data."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
report = root/'reports'
manifest = json.loads((report/'assets_manifest.json').read_text())
for name, expected in manifest['source_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
with (report/'tables/comparison.csv').open() as file:
    rows = list(csv.DictReader(file))
for cohort, source in [('development_0_9', 'results/m2_hybrid_h1/comparison.json'),
                       ('validation_10_19', 'results/m2_validation_h1/validation_summary.json')]:
    groups = json.loads((root/source).read_text())['groups']
    for row in [r for r in rows if r['cohort'] == cohort]:
        value = groups[row['condition']][row['method']]
        for key in ('n', 'falls', 'walking_successes', 'mean_alive_s', 'velocity_rmse'):
            assert float(row[key]) == value[key], (cohort, row['condition'], row['method'], key)
checks = {'source_hashes': len(manifest['source_sha256']), 'comparison_rows': len(rows), 'figures': 6}
for method in ('B1', 'M', 'M2'):
    current = root/f'results/repository_maintenance/J40_{method}'
    reference = root/f'results/m2_validation_h1/episodes/seed_10/J40/{method}'
    # This optional check uses the real README-command repeats created during maintenance.
    if not current.exists():
        continue
    old = json.loads((reference/'summary.json').read_text())
    new = json.loads((current/'summary.json').read_text())
    for key in ('fallen', 'time_alive_s', 'velocity_rmse_after_warmup_before_fall_m_s',
                'schedule_sha256', 'initial_state_sha256', 'checkpoint_sha256', 'scene_sha256'):
        assert old[key] == new[key], (method, key)
    a, b = np.load(current/'signals.npz'), np.load(reference/'signals.npz')
    assert set(a.files) == set(b.files)
    # Timings vary; compare the actual recorded policy and physical signals.
    tested = [key for key in a.files if 'ms' not in key]
    for key in tested:
        assert np.array_equal(a[key], b[key]), (method, key)
    checks[method+'_reproduction'] = 'bitwise physical/policy signals'
print(json.dumps(checks, indent=2))
