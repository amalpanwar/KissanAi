import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
from scripts import load_agmarknet_prices


class LoaderTests(unittest.TestCase):
    def test_mixed_historical_and_dashboard_schema(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'data/raw/live').mkdir(parents=True)
            pd.DataFrame([
                {'district_name':'Meerut','cmdt_name':'Wheat','model_price_wt':2400,'rep_date':'17-09-2026','as_on':None,'reported_date':None},
                {'district_name':'Meerut','cmdt_name':'Wheat','model_price_wt':None,'rep_date':None,'as_on':2500,'reported_date':'18-09-2026'},
            ]).to_csv(root/'data/raw/live/agmarknet_report.csv',index=False)
            cfg=SimpleNamespace(paths={'sqlite_db':str(root/'prices.db')})
            try:
                os.chdir(root)
                with patch.object(load_agmarknet_prices,'load_config',return_value=cfg):
                    load_agmarknet_prices.main()
            finally:
                os.chdir(original)
            with sqlite3.connect(root/'prices.db') as conn:
                rows=conn.execute('select modal_price,arrival_date from market_prices order by modal_price').fetchall()
            self.assertEqual(rows,[(2400.0,'17-09-2026'),(2500.0,'18-09-2026')])
