"""Regression for an active global grace period hiding an expired pool grace."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

path=Path(__file__).resolve().parents[1]/'scripts/check_push_t_storage_quota.py'
spec=importlib.util.spec_from_file_location('quota_guard',path)
quota=importlib.util.module_from_spec(spec); spec.loader.exec_module(quota)

class StorageQuotaGuardTests(unittest.TestCase):
    def test_expired_pool_blocks_even_when_global_grace_is_active(self):
        def command(args):
            if args[1]=='project': return '4101986 P /tmp\n'
            if args[1]=='getstripe': return 'pool: disk\npool: disk\n'
            if '-p' not in args: return ' 0 0 0 - 0 0 0 -\n'
            grace='expired' if '--pool' in args else '1w4d14h'
            return f'/nobackup/storage/disk\n 9424581088* 6291456000 15728640000 {grace} 4658744 0 0 -\n'
        with patch.object(quota,'read_command',side_effect=command):
            result=quota.check_storage('/tmp')
            self.assertFalse(result['allowed']); self.assertEqual(result['pools'],['disk'])
            self.assertEqual(len(result['violations']),1)
            self.assertIn('disk',result['violations'][0])
            with self.assertRaisesRegex(RuntimeError,'grace expired'): quota.require_storage('/tmp')

    def test_active_grace_below_hard_limit_is_accepted(self):
        self.assertEqual(quota.quota_violations(quota.parse_quota('900 600 1500 11d 900 1000 1500 -')),[])

    def test_hard_limit_blocks_without_grace_text(self):
        self.assertIn('hard limit',quota.quota_violations(quota.parse_quota('200 1 1 - 20 0 0 -'))[0])

    def test_expired_inode_quota_blocks(self):
        self.assertIn('files:',quota.quota_violations(quota.parse_quota('20 0 0 - 900 600 1500 expired'))[0])

    def test_unlimited_quotas_accept_usage(self):
        self.assertEqual(quota.quota_violations(quota.parse_quota('900 0 0 - 900 0 0 -')),[])

    def test_invalid_report_is_not_assumed_healthy(self):
        with self.assertRaises(ValueError): quota.parse_quota('quota service unavailable')


class StorageQuotaExceptionTests(unittest.TestCase):
    @staticmethod
    def command(global_grace='1w',pool_hard=1500,pool_files_grace='-',user_expired=False):
        def read(args):
            if args[1]=='project': return '4101986 P /tmp\n'
            if args[1]=='getstripe': return 'pool: disk\n'
            if '-u' in args and user_expired: return '900 600 1500 expired 0 0 0 -\n'
            if '-p' not in args: return '0 0 0 - 0 0 0 -\n'
            if '--pool' in args: return f'900 600 {pool_hard} expired 900 600 1500 {pool_files_grace}\n'
            return f'900 600 1500 {global_grace} 0 0 0 -\n'
        return read

    def test_explicit_exception_requires_successful_live_probe(self):
        with patch.object(quota,'read_command',side_effect=self.command()), patch.object(quota,'probe_output_writes',return_value=[{'verified':True}]) as probe:
            result=quota.require_storage('/tmp',allow_expired_disk_pool_after_write_probe=True)
        self.assertTrue(result['allowed']); self.assertEqual(len(result['waived_violations']),1)
        self.assertEqual(len(result['original_violations']),1); probe.assert_called_once()

    def test_exception_never_accepts_a_failed_actual_write(self):
        with patch.object(quota,'read_command',side_effect=self.command()), patch.object(quota,'probe_output_writes',side_effect=OSError(122,'Disk quota exceeded')):
            with self.assertRaisesRegex(RuntimeError,'Actual output write probe failed'):
                quota.require_storage('/tmp',allow_expired_disk_pool_after_write_probe=True)

    def test_global_hard_inode_and_user_limits_cannot_be_waived(self):
        for kwargs in ({'global_grace':'expired'},{'pool_hard':900},{'pool_files_grace':'expired'},{'user_expired':True}):
            with self.subTest(kwargs=kwargs), patch.object(quota,'read_command',side_effect=self.command(**kwargs)), patch.object(quota,'probe_output_writes') as probe:
                result=quota.check_storage('/tmp',allow_expired_disk_pool_after_write_probe=True)
                self.assertFalse(result['allowed']); probe.assert_not_called()

    def test_without_explicit_opt_in_default_remains_strict(self):
        with patch.object(quota,'read_command',side_effect=self.command()), patch.object(quota,'probe_output_writes') as probe:
            result=quota.check_storage('/tmp')
            self.assertFalse(result['allowed']); probe.assert_not_called()

if __name__=='__main__': unittest.main()
