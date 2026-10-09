import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from check_sensor_age import audit_episode


def main():
    root = Path('results/m2_validation_h1')
    audit_episode(root/'cli_check')
    with np.load(root/'cli_check/signals.npz') as cli, np.load(root/'episodes/seed_10/J40/M2/signals.npz') as reference:
        for k in ('obs', 'actions', 'q_true', 'dq_true', 'g_true', 'omega_true'):
            np.testing.assert_array_equal(cli[k], reference[k])
    cli = json.loads((root/'cli_check/summary.json').read_text())
    assert cli['adapter_config']['family'] == 'hybrid'
    assert (cli['horizon_joints_ms'], cli['horizon_imu_ms']) == (120, 20)
    a = json.loads((root/'interactive_checks/summary.json').read_text())
    b = json.loads((root/'window_check/summary.json').read_text())
    for summary in (a, b):
        assert summary['compensator'] == 'm2' and summary['reset_count'] == 1
        assert summary['status'] == 'completed' and summary['fall_time_s'] is None
        assert summary['m2_profile_sha256'] == a['m2_profile_sha256']
    with np.load(root/'interactive_checks/chunks/000000.npz') as headless, np.load(root/'window_check/chunks/000000.npz') as window:
        count = len(window['obs'])
        for k in ('obs', 'actions', 'qpos', 'qvel', 'ages_s'):
            np.testing.assert_array_equal(headless[k][:count], window[k])
        assert count == 800
    print(json.dumps({'cli_default_profile_and_trace': 'passed',
                      'interactive_C_W_R_and_reset': 'passed',
                      'gui_headless_bitwise_equivalence': 'passed',
                      'window_simulation_seconds': b['simulation_seconds'],
                      'window_wall_seconds': b['wall_seconds']}, indent=2))


if __name__ == '__main__':
    main()
