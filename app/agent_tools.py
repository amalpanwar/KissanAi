"""Grounded tool adapters used by the specialist agents."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from app.agent_system import AgentResult
from app.location_lookup import lookup_place_in_text

ROOT = Path(__file__).resolve().parents[1]
MARKET_CSV = ROOT / "data/raw/live/agmarknet_report.csv"


class AdvisorTools:
    def __init__(self, advisor):
        self.advisor = advisor

    def location(self, payload):
        loc = lookup_place_in_text(payload["question"])
        if loc:
            return loc
        match = re.search(r"(?:District|जिला):\s*([^|]+)", payload.get("context", ""), re.I)
        if match:
            value = match.group(1).strip()
            return lookup_place_in_text(value) or {"district": value, "place": value}
        return {}

    def weather(self, goal, payload):
        from app.advisor import WeatherRequest
        question = payload["question"]
        normalized = self.advisor._normalize_hinglish(question)
        request = self.advisor._parse_weather_request(question, normalized) or WeatherRequest()
        loc = self.location(payload)
        if not request.place and not loc:
            return AgentResult("मौसम के लिए अपना गांव/शहर या जिला बताएं।", "needs_input")
        result = self.advisor._answer_weather_request(request, question, normalized,
                       place_override=request.place or loc.get("place") or loc.get("district"))
        answer = result["answer"]
        status = "ok" if result.get("references") else "unavailable"
        if "पिछले उपलब्ध अपडेट" in answer:
            status = "stale"
        return AgentResult(answer, status, result.get("references", []),
                           {"checked_at": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()},
                           {"weather_action": result.get("weather_action")})

    def refresh_market(self, goal, payload):
        from scripts.agmarknet_daily_refresh import _latest_report_date
        today = datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat()
        latest = _latest_report_date(MARKET_CSV)
        if latest != today:
            env = os.environ.copy()
            env.update({"AGMARKNET_MODE": "dashboard", "AGMARKNET_LOOKBACK_DAYS": "3", "AGMARKNET_ALL_DISTRICTS": "1"})
            # The refresh has its own inter-process lock. Killing the parent on timeout
            # would orphan its fetch child, so the timeout belongs to the refresh.
            env["AGMARKNET_RUN_TIMEOUT_SEC"] = os.getenv("KISAANAI_QUERY_REFRESH_TIMEOUT", "45")
            meta = ROOT / "data/raw/live/agmarknet_refresh_status.json"
            recent_attempt = meta.exists() and (datetime.now().timestamp() - meta.stat().st_mtime) < 1800
            if not recent_attempt:
                subprocess.run([sys.executable, str(ROOT / "scripts/agmarknet_daily_refresh.py")],
                               cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                latest = _latest_report_date(MARKET_CSV)
        return AgentResult("Market freshness checked", "ok" if latest == today else "stale",
                           evidence={"latest_report_date": latest, "checked_date": today})

    def prices(self, goal, payload):
        question = payload["question"]
        msp = self.advisor._answer_msp_query(question)
        if msp:
            return AgentResult(msp, references=["PIB MSP notification"])
        district = self.location(payload).get("district")
        if not district:
            return AgentResult("मंडी भाव के लिए अपना जिला बताएं।", "needs_input")
        if not MARKET_CSV.exists():
            return AgentResult("अभी Agmarknet का सत्यापित मूल्य डेटा उपलब्ध नहीं है।", "unavailable")
        df = pd.read_csv(MARKET_CSV, low_memory=False)
        needed = {"district_name", "cmdt_name", "rep_date", "model_price_wt"}
        if not needed.issubset(df.columns):
            return AgentResult("Agmarknet डेटा का प्रारूप पढ़ा नहीं जा सका।", "unavailable")
        df = df[df["district_name"].astype(str).str.casefold() == str(district).casefold()].copy()
        normalized = self.advisor._normalize_hinglish(question)
        crop = self.advisor._extract_crop_from_query(normalized)
        from app.advisor import WESTERN_UP_CROP_BASELINES
        aliases = WESTERN_UP_CROP_BASELINES.get(crop, {}).get("market_aliases", [crop] if crop else [])
        if not aliases:
            aliases = [c for c in df["cmdt_name"].dropna().unique() if str(c).casefold() in question.casefold()]
        if aliases:
            mask = pd.Series(False, index=df.index)
            for alias in aliases:
                mask |= df["cmdt_name"].astype(str).str.contains(str(alias), case=False, regex=False)
            df = df[mask].copy()
        elif not re.search(r"latest|all|सभी|ताजा|ताज़ा|आज.*भाव", question, re.I):
            return AgentResult("किस फसल/कमोडिटी का मंडी भाव चाहिए?", "needs_input")
        df["date"] = pd.to_datetime(df["rep_date"], format="mixed", dayfirst=True, errors="coerce")
        df["price"] = pd.to_numeric(df["model_price_wt"], errors="coerce")
        today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
        df = df.dropna(subset=["date", "price"])
        df = df[(df["date"].dt.date <= today) & (df["price"] > 0)]
        if df.empty:
            return AgentResult(f"{district} में इस कमोडिटी का सत्यापित Agmarknet भाव नहीं मिला।", "unavailable")
        keys = [c for c in ["cmdt_name", "market_name"] if c in df.columns]
        rows = df.sort_values("date").groupby(keys, as_index=False).tail(1).sort_values("date", ascending=False).head(10)
        recent = rows[rows["date"].dt.date.map(lambda d: (today - d).days <= 3)]
        stale = recent.empty
        if not recent.empty:
            rows = recent
        lines = [f"{district} — Agmarknet में उपलब्ध मंडी भाव:"]
        evidence = []
        for _, row in rows.iterrows():
            date = row["date"].date().isoformat()
            unit = row.get("unit_name_price")
            unit = str(unit) if pd.notna(unit) else "इकाई उपलब्ध नहीं"
            lines.append(f"- {row['cmdt_name']} | {row.get('market_name', district)}: {row['price']:,.2f} {unit} | {date}")
            evidence.append({"commodity": row["cmdt_name"], "market": row.get("market_name", ""), "date": date, "price": float(row["price"]), "unit": unit})
        if stale:
            lines.append("पुराने रिकॉर्ड दिखाए गए हैं; इन्हें आज का भाव न मानें।")
        return AgentResult("\n".join(lines), "stale" if stale else "ok", ["https://agmarknet.gov.in/"], {"records": evidence})

    def pesticides(self, goal, payload):
        q = self.advisor._normalize_hinglish(payload["question"])
        request = self.advisor._parse_pesticide_request(payload["question"], q, payload.get("context", ""))
        result = self.advisor._structured_pesticide_advice(q,
                 crop_hint=self.advisor._extract_preferred_crop_from_context(payload.get("context", "")), request=request)
        refs = result.get("references", [])
        return AgentResult(result["answer"], "ok" if refs else "needs_input", refs)

    def agronomy(self, goal, payload):
        text = f"{payload['context']} किसान का प्रश्न: {payload['question']}" if payload.get("context") else payload["question"]
        result = self.advisor._answer_legacy(text)
        return AgentResult(result["answer"], references=result.get("references", []),
                           metadata={"topic": result.get("topic", "rag")})
