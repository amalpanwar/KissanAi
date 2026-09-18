import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

import pandas as pd
from app.agmarknet_client import _fetch_report_page
from scripts import agmarknet_fetch


class RefreshTests(unittest.TestCase):
    def test_http_400_is_not_silently_no_data(self):
        err = HTTPError("https://example.test", 400, "Bad Request", {}, io.BytesIO(b'{"message":"Invalid filter"}'))
        with patch("app.agmarknet_client.urlopen", side_effect=err):
            with self.assertRaises(HTTPError):
                _fetch_report_page({}, retries=1)

    def run_fetch(self, root, side_effect, max_pages="5"):
        argv = ["fetch", "--mode", "dashboard", "--state_ids", "34", "--from_date", "2026-09-18", "--to_date", "2026-09-18",
                "--out", str(root / "prices.csv"), "--fail_log", str(root / "fail.csv"), "--summary_json", str(root / "status.json"),
                "--merge_existing", "--require_complete", "--max_pages", max_pages, "--sleep_sec", "0"]
        with patch("sys.argv", argv), patch.object(agmarknet_fetch, "fetch_page", side_effect=side_effect):
            return agmarknet_fetch.main()

    def page(self, price, pages=1):
        return {"pagination": {"total_pages": pages, "current_page": 1}, "data": {"records": [
            {"state_name": "Uttar Pradesh", "district_name": "Meerut", "market_name": "Meerut", "cmdt_name": "Wheat", "rep_date": "18-09-2026", "model_price_wt": price}]}}

    def test_partial_fetch_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "prices.csv").write_text("original bytes\n")
            rc = self.run_fetch(root, [self.page(2500, 2), RuntimeError("network failed")])
            self.assertEqual(rc, 5)
            self.assertEqual((root / "prices.csv").read_text(), "original bytes\n")

    def test_page_limit_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(self.run_fetch(root, [self.page(2500, 2)], "1"), 5)
            self.assertFalse((root / "prices.csv").exists())

    def test_corrected_price_replaces_same_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame(self.page(2000)["data"]["records"]).to_csv(root / "prices.csv", index=False)
            self.assertEqual(self.run_fetch(root, [self.page(2500)]), 0)
            frame = pd.read_csv(root / "prices.csv")
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.iloc[0]["model_price_wt"], 2500)

if __name__ == "__main__":
    unittest.main()
