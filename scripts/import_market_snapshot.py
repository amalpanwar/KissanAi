"""Import an official CSV batch without deleting or rewriting historical rows."""
from __future__ import annotations
import argparse
import csv
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def merge_snapshot(existing: Path, snapshot: Path) -> int:
    with snapshot.open(newline='') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        new_rows = list(reader)
    required = {'state_name', 'district_name', 'cmdt_name', 'rep_date', 'model_price_wt'}
    if not required.issubset(fields) or not new_rows:
        raise ValueError('Snapshot is empty or missing required columns')
    today = datetime.now(ZoneInfo('Asia/Kolkata')).date()
    for row in new_rows:
        date = datetime.strptime(row['rep_date'], '%d-%m-%Y').date()
        if date > today or float(row['model_price_wt']) <= 0:
            raise ValueError('Snapshot contains future dates or invalid prices')
    key_fields = ['state_name','district_name','market_name','cmdt_name','rep_date','variety_name','grade_name','model_price_wt']
    def key(row):
        return tuple(str(row.get(k) or '') for k in key_fields)
    seen = set()
    if existing.exists():
        with existing.open(newline='') as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or fields
            seen = {key(row) for row in reader}
    added = []
    for row in new_rows:
        if key(row) not in seen:
            added.append(row)
            seen.add(key(row))
    if not added:
        return 0
    existing.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=existing.parent, suffix='.csv')
    os.close(fd)
    temp = Path(name)
    try:
        if existing.exists():
            shutil.copyfile(existing, temp)
        with temp.open('a', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
            if not existing.exists():
                writer.writeheader()
            else:
                with existing.open('rb') as source:
                    source.seek(-1, 2)
                    if source.read(1) != b'\n':
                        handle.write('\n')
            writer.writerows(added)
        temp.replace(existing)
    finally:
        temp.unlink(missing_ok=True)
    return len(added)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('snapshot', type=Path)
    args = parser.parse_args()
    from agmarknet_daily_refresh import OUT_CSV, STATUS_JSON, _latest_report_date, _write_catalog
    STATUS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with STATUS_JSON.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        added = merge_snapshot(OUT_CSV, args.snapshot)
        _write_catalog(OUT_CSV)
        subprocess.run([sys.executable, str(ROOT/'scripts/load_agmarknet_prices.py')], cwd=ROOT, check=True)
        status = {'status':'success','message':'Imported validated official snapshot and synchronized SQLite.',
                  'after_latest':_latest_report_date(OUT_CSV),'new_rows':added,
                  'updated_at':datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(), 'mode':'snapshot_import'}
        temp = STATUS_JSON.with_suffix('.tmp')
        temp.write_text(json.dumps(status,indent=2))
        temp.replace(STATUS_JSON)
        print(f'Imported {added} new records')


if __name__ == '__main__':
    main()
