"""Regression coverage for resumed-run provenance and reward diagnostics."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/summarize_push_t_weight_ablation.py'
spec = importlib.util.spec_from_file_location('weight_ablation_summary', SCRIPT)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def batch(step, chunk, value=0.1, uids=(17,)):
    return {
        'global_step': step,
        'metadata': {'rollout_uid': [[uid] for uid in uids],
                     'chunk_id': [[chunk] for _ in uids]},
        'trajectory_view_mse': {view: [value]*len(uids) for view in report.WEIGHTS['original']},
        'done': [chunk == 12]*len(uids),
        'terminal_view_mse_done': {view: [value]*len(uids) if chunk == 12 else []
                                   for view in report.WEIGHTS['original']},
    }


class SummaryRegression(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self):
        out = self.root/'comparison'
        out.mkdir()
        submissions = []
        for arm, job in [('original', '102'), ('no_wrist', '103')]:
            submissions.append({'arm': arm, 'job_id': job, 'segment_start': 0, 'segment_end': 5})
            run = self.root/'outputs'/f'push_t_combined_mse_fp32_h13_{arm}_15u_s5'
            records = [{'update': 0, 'rollout_uid': uid, 'group_id': uid//16,
                        'valid': True, 'model_success': uid < 2,
                        'normalized_advantage': -1.0 if uid < 2 else 0.0,
                        'chunk_advantages': ([0.5]*13 if uid == 0 else [-0.25]*13),
                        'trajectory_mse': 0.1, 'lastframe_mse': 0.1}
                       for uid in range(128)]
            write_jsonl(run/'trajectory_records/update_0000.jsonl', records)
            # The obsolete retry sorts after the current one and has every chunk.
            write_jsonl(run/'reward_view_records/rank_000_job_999.jsonl',
                        [batch(0, c, value=9.0, uids=range(128)) for c in range(13)])
            write_jsonl(run/f'reward_view_records/rank_000_job_{job}.jsonl',
                        [batch(0, c, uids=range(128)) for c in range(13)])
        (out/'submission_records.json').write_text(json.dumps(submissions))
        return out

    def test_active_chain_uses_only_its_own_segment_sidecars(self):
        run = self.root
        write_jsonl(run/'reward_view_records/rank_000_job_101.jsonl',
                    [batch(4, 0, value=0.4), batch(5, 0, value=8.0)])
        write_jsonl(run/'reward_view_records/rank_000_job_102.jsonl',
                    [batch(5, 0, value=0.5), batch(4, 0, value=8.0)])
        write_jsonl(run/'reward_view_records/rank_999.jsonl', [batch(4, 0, value=9.0)])
        write_jsonl(run/'reward_view_records/rank_999_job_999.jsonl', [batch(5, 0, value=9.0)])
        samples = report.sidecars(run, {'101': (0, 5), '102': (5, 10)})
        self.assertEqual(set(samples), {(4, 17), (5, 17)})
        self.assertEqual(samples[(4, 17)][0]['trajectory']['main'], 0.4)
        self.assertEqual(samples[(5, 17)][0]['trajectory']['main'], 0.5)

    def test_missing_current_chunk_cannot_be_filled_from_an_old_retry(self):
        out = self.fixture()
        run = self.root/'outputs/push_t_combined_mse_fp32_h13_original_15u_s5'
        write_jsonl(run/'reward_view_records/rank_000_job_102.jsonl',
                    [batch(0, c, uids=range(128)) for c in range(12)])
        with patch.object(report, 'ROOT', self.root):
            result = report.summarize(out)
        self.assertEqual(result['arms']['original']['training'][0]['complete_per_view_records'], 0)
        self.assertIn('original: 0 has only 0/128 per-view records', result['integrity_errors'])
        self.assertFalse(result['complete'])

    def test_summary_distinguishes_full_return_and_actual_chunk_advantages(self):
        out = self.fixture()
        with patch.object(report, 'ROOT', self.root):
            result = report.summarize(out)
        self.assertEqual(result['integrity_errors'], [])
        for arm in report.WEIGHTS:
            row = result['arms'][arm]['training'][0]
            self.assertEqual(row['complete_per_view_records'], 128)
            self.assertLess(row['max_mse_recompute_error'], 1e-12)
            self.assertEqual(row['success_negative_advantage'], 2)
            self.assertEqual(row['success_negative_mean_chunk_advantage'], 1)
            self.assertEqual(row['success_negative_chunk_fraction'], 0.5)
            self.assertEqual(row['success_mean_chunk_advantage'], 0.125)
        self.assertFalse(result['complete'])

    def test_legacy_run_without_submission_manifest_remains_readable(self):
        self.assertIsNone(report.submitted_job_ranges(self.root, 'original'))
        write_jsonl(self.root/'reward_view_records/rank_000.jsonl', [batch(0, 0)])
        self.assertEqual(set(report.sidecars(self.root)), {(0, 17)})


if __name__ == '__main__':
    unittest.main()
