"""Storage failures must preserve optional diagnostics without killing rollouts."""
import errno
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import importlib.util
spec = importlib.util.spec_from_file_location(
    'diagnostic_io_under_test',
    Path(__file__).resolve().parents[1] / 'third_party/rlinf_runtime/rlinf/utils/diagnostic_io.py',
)
io = importlib.util.module_from_spec(spec)
spec.loader.exec_module(io)


class DiagnosticIoTests(unittest.TestCase):
    def test_shared_quota_failure_preserves_local_and_recovers_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            shared = base/'shared/rank_000_job_1.jsonl'
            local = base/'local/rank_000_job_1.jsonl'
            spool = io.DiagnosticJsonlSpool(shared, local)
            real_write = io.atomic_write
            def quota_write(path, payload):
                if path == shared:
                    raise OSError(errno.EDQUOT, 'synthetic quota failure')
                return real_write(path, payload)
            with patch.object(io, 'atomic_write', side_effect=quota_write):
                self.assertFalse(spool.append({'chunk':0}, publish=True))
                self.assertFalse(spool.append({'chunk':1}, publish=True))
            self.assertEqual([json.loads(line) for line in local.read_text().splitlines()], [{'chunk':0},{'chunk':1}])
            self.assertFalse(shared.exists())
            self.assertTrue(spool.publish())
            self.assertEqual(shared.read_bytes(), local.read_bytes())
            self.assertEqual(io.publish_directory(local.parent, shared.parent), 0)
            self.assertEqual(len(shared.read_text().splitlines()), 2)

    def test_both_disks_failing_keeps_memory_for_later_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary); spool=io.DiagnosticJsonlSpool(base/'shared/log.jsonl',base/'local/log.jsonl')
            with patch.object(io,'atomic_write',side_effect=OSError(errno.ENOSPC,'synthetic full disk')):
                self.assertFalse(spool.append({'chunk':0},publish=True))
            self.assertTrue(spool.append({'chunk':1},publish=True))
            self.assertEqual(len(spool.destination.read_text().splitlines()),2)
            with self.assertRaises(ValueError):
                spool.append({'mse':float('nan')},publish=True)

    def test_shared_probe_exercises_reopened_writes_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            result=io.probe_directory(directory,workers=4)
            self.assertEqual(result['bytes_verified'],4*3*65536)
            self.assertEqual(list(directory.iterdir()),[])


if __name__ == '__main__':
    unittest.main()
