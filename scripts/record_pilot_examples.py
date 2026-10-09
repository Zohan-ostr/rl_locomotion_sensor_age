"""Replay seed 0 J80 for visual fall-rule inspection; exclude replays from N."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_sensor_episode import configure_threads, run_episode
from check_sensor_age import audit_episode

import imageio.v2 as imageio
import numpy as np


def main():
    root = Path('results/pilot_h1')
    selection = json.loads((root/'selection.json').read_text())
    horizons = [selection['selected']['joints_ms'], selection['selected']['imu_ms']]
    configure_threads(1)
    results = []
    for method in ('B1', 'M'):
        output = root/f'visual_checks/seed0_J80_{method}'
        h = horizons[0]
        source = root/f'episodes/seed_0/J80/{method}_H{h}_{h}' if method == 'M' else root/'episodes/seed_0/J80/B1_H0_0'
        summary = run_episode(output, root/'inputs/seed_0/J80.json', root/'inputs/seed_0/initial.json',
                              method, tuple(horizons) if method == 'M' else (0, 0), video=True)
        audit_episode(output)
        with np.load(source/'signals.npz') as a, np.load(output/'signals.npz') as b:
            for key in ('obs', 'actions', 'q_true', 'dq_true', 'g_true', 'omega_true'):
                np.testing.assert_array_equal(a[key], b[key])
        original = json.loads((source/'summary.json').read_text())
        assert summary['fall_time_s'] == original['fall_time_s']
        reader = imageio.get_reader(output/'walk.mp4')
        frame_count = reader.count_frames()
        indices = [max(0, frame_count-26), max(0, frame_count-6), frame_count-1]
        imageio.imwrite(output/'fall_frames.png', np.concatenate([reader.get_data(i) for i in indices], axis=1))
        metadata = reader.get_meta_data(); reader.close()
        record = {'method': method, 'source': str(source), 'visual_replay': str(output),
                  'physical_and_policy_trajectory_bitwise_equal': True, 'fall_time_s': summary['fall_time_s'],
                  'frame_count': frame_count, 'frame_indices': indices, 'fps': metadata['fps'],
                  'frames_at_simulation_time_s': [(i+1)/metadata['fps'] for i in indices],
                  'loop_wall_seconds_with_rendering': summary['wall_seconds_loop'],
                  'excluded_from_episode_counts': True}
        results.append(record)
        print(json.dumps(record), flush=True)
    (root/'visual_checks/checks.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
