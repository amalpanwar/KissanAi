import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts import agmarknet_daily_refresh as refresh


class RefreshRecoveryTests(unittest.TestCase):
    def test_failed_round_retries_and_publishes_only_after_success(self):
        with tempfile.TemporaryDirectory() as directory:
            status=Path(directory)/'status.json'
            summary=status.with_name('agmarknet_fetch_summary.json')
            calls=[]
            def run(cmd, **kwargs):
                calls.append(cmd)
                if 'agmarknet_fetch.py' in cmd[1]:
                    success=len(calls)==3
                    summary.write_text(json.dumps({'status':'success' if success else 'failed','failure_count':0 if success else 1}))
                    return 0 if success else 5
                return 0
            today=refresh.datetime.now(refresh.ZoneInfo('Asia/Kolkata')).date().isoformat()
            with patch.object(refresh,'STATUS_JSON',status), patch.object(refresh,'_latest_report_date',return_value=today), patch.object(refresh,'_write_catalog') as catalog, patch.object(refresh.subprocess,'call',side_effect=run), patch.object(refresh.time,'sleep') as sleep, patch.dict(os.environ,{'AGMARKNET_MODE':'auto','AGMARKNET_REFRESH_ATTEMPTS':'2'}):
                self.assertEqual(refresh._refresh(),0)
            self.assertEqual(len(calls),4)
            catalog.assert_called_once()
            sleep.assert_called_once_with(10)
            self.assertEqual(len(json.loads(status.read_text())['attempts']),3)

    def test_exhaustion_keeps_failure_and_never_loads_database(self):
        with tempfile.TemporaryDirectory() as directory:
            status=Path(directory)/'status.json'
            with patch.object(refresh,'STATUS_JSON',status), patch.object(refresh,'_latest_report_date',return_value='2026-10-11'), patch.object(refresh,'_write_catalog') as catalog, patch.object(refresh.subprocess,'call',return_value=5) as run, patch.object(refresh.time,'sleep'), patch.dict(os.environ,{'AGMARKNET_MODE':'auto','AGMARKNET_REFRESH_ATTEMPTS':'2'}):
                self.assertEqual(refresh._refresh(),5)
            self.assertEqual(run.call_count,4)
            catalog.assert_not_called()
            self.assertEqual(json.loads(status.read_text())['status'],'failed')
