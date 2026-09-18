import csv
import tempfile
import unittest
from pathlib import Path
from scripts.import_market_snapshot import merge_snapshot


class SnapshotTests(unittest.TestCase):
    def test_history_preserved_and_import_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fields = ['state_name','district_name','market_name','cmdt_name','rep_date','model_price_wt']
            old = 'state_name,district_name,market_name,cmdt_name,rep_date,model_price_wt\nUttar Pradesh,Meerut,Meerut,Wheat,13-05-2026,2400\n'
            existing=root/'existing.csv'; existing.write_text(old)
            snapshot=root/'new.csv'
            with snapshot.open('w',newline='') as handle:
                writer=csv.writer(handle);writer.writerow(fields)
                writer.writerow(['Uttar Pradesh','Meerut','Meerut','Wheat','18-09-2026',2500])
            self.assertEqual(merge_snapshot(existing,snapshot),1)
            self.assertTrue(existing.read_text().startswith(old))
            first=existing.read_bytes()
            self.assertEqual(merge_snapshot(existing,snapshot),0)
            self.assertEqual(existing.read_bytes(),first)

    def test_bad_snapshot_cannot_change_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);existing=root/'old.csv';existing.write_text('keep me')
            snapshot=root/'new.csv';snapshot.write_text('bad\ninput\n')
            with self.assertRaises(ValueError):merge_snapshot(existing,snapshot)
            self.assertEqual(existing.read_text(),'keep me')
