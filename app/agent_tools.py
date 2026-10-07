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
from app.location_selection import location_from_context, qualified_place, scope_market_rows, market_scope_caption

ROOT = Path(__file__).resolve().parents[1]
MARKET_CSV = ROOT / "data/raw/live/agmarknet_report.csv"


class AdvisorTools:
    def __init__(self, advisor):
        self.advisor = advisor

    def location(self, payload):
        selected = location_from_context(payload.get("context", ""))
        if selected.get("invalid_selection"):
            return selected
        # The full UI hierarchy is authoritative. Explicit query locations have
        # already been resolved within that district before context is composed.
        if selected.get("place"):
            return selected
        loc = lookup_place_in_text(payload["question"], state=selected.get("state", ""), district=selected.get("district", ""))
        return loc or selected

    def weather(self, goal, payload):
        question = payload["question"]
        normalized = self.advisor._normalize_hinglish(question)
        request = self.advisor._parse_weather_request(question, normalized)
        if request is None:
            from app.advisor import WeatherRequest
            request = WeatherRequest()
        loc = self.location(payload)
        if loc.get("invalid_selection"):
            return AgentResult("स्थान चयन अब उपलब्ध नहीं है। कृपया गांव/कस्बा फिर चुनें।", "needs_input")
        if not request.place and not loc:
            return AgentResult("मौसम के लिए अपना गांव/शहर या जिला बताएं।", "needs_input")
        if loc:
            from app.weather import (get_current_weather_hindi, get_daily_weather_forecast_hindi,
                                     get_weekly_weather_forecast_hindi, get_rain_day_forecast_hindi)
            place = qualified_place(loc)
            if request.action == "rain_day":
                answer = get_rain_day_forecast_hindi(place, days=7)
            elif request.action == "weekly":
                answer = get_weekly_weather_forecast_hindi(place, days=7)
            elif request.action in {"daily", "daily_rain"} and request.day_offset is not None:
                answer = get_daily_weather_forecast_hindi(place, day_offset=request.day_offset,
                                                         label=request.label, rain_focus=request.action == "daily_rain")
            else:
                answer = get_current_weather_hindi(place)
            result = {"answer": answer, "references": ["Open-Meteo API"], "weather_action": request.action}
        else:
            result = self.advisor._answer_weather_request(request, question, normalized, place_override=request.place)
        answer = result["answer"]
        status = "ok" if result.get("references") else "unavailable"
        if "नहीं मिल पाया" in answer:
            status = "unavailable"
            result["references"] = []
        if "पिछले उपलब्ध अपडेट" in answer:
            status = "stale"
        return AgentResult(answer, status, result.get("references", []),
                           {"checked_at": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(), "location": loc},
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
        from app.commodity_lookup import (catalog_names, resolve_commodities,
                                          matching_market_names, normalize)
        question = payload["question"]
        location = self.location(payload)
        if location.get("invalid_selection"):
            return AgentResult("स्थान चयन अब उपलब्ध नहीं है। कृपया गांव/कस्बा फिर चुनें।", "needs_input")
        requested = resolve_commodities(question, catalog_names())
        # Preserve the narrow built-in aliases if a deployment lacks the dictionary.
        crop = self.advisor._extract_crop_from_query(self.advisor._normalize_hinglish(question))
        if not requested and crop:
            requested = [crop]
        cane = None
        if any(normalize(name) == "sugarcane" for name in requested):
            from app.sugarcane_prices import sugarcane_price_result
            cane = sugarcane_price_result(location)
            requested = [name for name in requested if normalize(name) != "sugarcane"]
            if not requested:
                return cane

        def finish(result):
            if cane is None:
                return result
            return AgentResult(cane.answer + "\n\n" + result.answer,
                               "ok" if cane.status == result.status == "ok" else "partial",
                               list(dict.fromkeys(cane.references + result.references)),
                               {**result.evidence, "sugarcane": cane.evidence})

        msp = self.advisor._answer_msp_query(question)
        if msp and cane is None:
            return AgentResult(msp, references=["PIB MSP notification"])
        district = location.get("district")
        if not district:
            return finish(AgentResult("मंडी भाव के लिए अपना जिला बताएं।", "needs_input"))
        try:
            df = pd.read_csv(MARKET_CSV, low_memory=False)
        except (OSError, ValueError, pd.errors.ParserError):
            return finish(AgentResult("Agmarknet मूल्य फ़ाइल अभी उपलब्ध नहीं है या पढ़ी नहीं जा सकी। कृपया थोड़ी देर बाद प्रयास करें।", "unavailable"))
        needed = {"state_name", "district_name", "cmdt_name", "rep_date", "model_price_wt"}
        if not needed.issubset(df.columns):
            return finish(AgentResult("Agmarknet डेटा का प्रारूप पढ़ा नहीं जा सका।", "unavailable"))
        names = df["cmdt_name"].dropna().astype(str).unique().tolist()
        # Also recognize newly reported commodities not yet in the saved catalog.
        requested = list(dict.fromkeys(requested + resolve_commodities(question, names)))
        requested = [name for name in requested if normalize(name) != "sugarcane"]
        if requested:
            matches = matching_market_names(requested, names)
            df = df[df["cmdt_name"].isin(matches)].copy()
        elif not re.fullmatch(
            r"(?:show |tell me )?(?:all (?:commodity |crop |mandi )?prices|latest (?:commodity |crop |mandi )?prices)\??|"
            r"(?:आज के |ताजा |ताज़ा )?(?:सभी (?:फसलों के )?|मंडी के )भाव(?: बताएं| बताओ)?[?।]?",
            question.strip(), re.I,
        ):
            return finish(AgentResult("किस फसल/कमोडिटी का भाव चाहिए? उसका नाम लिखें, जैसे टमाटर, प्याज या गेहूं। नाम नहीं पहचान पाने पर दूसरी फसल का भाव नहीं दिखाया जाएगा।", "needs_input"))
        df, market_scope = scope_market_rows(df, location)
        df["date"] = pd.to_datetime(df["rep_date"], format="mixed", dayfirst=True, errors="coerce", utc=True)
        df["price"] = pd.to_numeric(df["model_price_wt"], errors="coerce")
        today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
        df = df.dropna(subset=["date", "price", "cmdt_name"])
        df = df[(df["date"].dt.date <= today) & (df["price"] > 0) & (df["price"] < float("inf"))]
        if df.empty:
            label = ", ".join(requested) or "मांगी गई कमोडिटी"
            return finish(AgentResult(f"{district} में {label} का सत्यापित Agmarknet भाव नहीं मिला। किसी दूसरी फसल का भाव नहीं दिखाया गया है।", "unavailable"))
        keys = [c for c in ["cmdt_name", "market_name"] if c in df.columns]
        rows = df.sort_values("date").groupby(keys, as_index=False, dropna=False).tail(1).sort_values("date", ascending=False)
        # Keep each requested commodity, even when another has newer records.
        rows = rows.groupby("cmdt_name", sort=False, dropna=False).head(3).head(30)
        lines = [market_scope_caption(location, market_scope), f"{district} — Agmarknet में उपलब्ध मंडी भाव:"]
        evidence = []
        for _, row in rows.iterrows():
            date = row["date"].date().isoformat()
            unit = row.get("unit_name_price")
            unit = str(unit) if pd.notna(unit) else "इकाई उपलब्ध नहीं"
            market = row.get("market_name")
            market = str(market).strip() if pd.notna(market) and str(market).strip() else "मंडी का नाम उपलब्ध नहीं (जिला रिकॉर्ड)"
            is_stale = (today - row["date"].date()).days > 3
            suffix = " — पुराना रिकॉर्ड, आज का भाव नहीं" if is_stale else ""
            lines.append(f"- {row['cmdt_name']} | {market}: {row['price']:,.2f} {unit} | {date}{suffix}")
            evidence.append({"commodity": row["cmdt_name"], "market": market, "date": date, "price": float(row["price"]), "unit": unit, "stale": is_stale})
        absent = [name for name in requested if not matching_market_names([name], rows["cmdt_name"].unique())]
        if absent:
            lines.append("इनके लिए इस स्थान का सत्यापित भाव नहीं मिला: " + ", ".join(absent))
        stale = any(row["stale"] for row in evidence)
        return finish(AgentResult("\n".join(lines), "partial" if absent else "stale" if stale else "ok",
                           ["https://agmarknet.gov.in/"], {"records": evidence, "location": location, "market_scope": market_scope}))

    def pesticides(self, goal, payload):
        q = self.advisor._normalize_hinglish(payload["question"])
        request = self.advisor._parse_pesticide_request(payload["question"], q, payload.get("context", ""))
        result = self.advisor._structured_pesticide_advice(q,
                 crop_hint=self.advisor._extract_preferred_crop_from_context(payload.get("context", "")), request=request)
        refs = result.get("references", [])
        return AgentResult(result["answer"], "ok" if refs else "needs_input", refs)

    def news(self, goal, payload):
        from app.agriculture_news import get_news, relevant_articles, format_news
        related = bool(payload.get("related_only"))
        # The sidebar refreshes the shared feed. Optional related news must never
        # delay an existing crop guide or make a model-less answer depend on HTTP.
        snapshot = get_news(cache_only=related)
        articles = relevant_articles(snapshot["articles"], payload["question"], related_only=related)
        return AgentResult(format_news(snapshot, articles, related_only=related), snapshot["status"],
                           [item["url"] for item in articles],
                           {"articles": articles, "fetched_at": snapshot["fetched_at"], "status": snapshot["status"]},
                           {"topic": "news"})

    def agronomy(self, goal, payload):
        question = payload["question"]
        if self.advisor._is_crop_guide_followup_intent(question):
            from app.crop_guide import build_crop_production_followup
            crop = (self.advisor._extract_crop_from_query(self.advisor._normalize_hinglish(question))
                    or self.advisor._extract_preferred_crop_from_context(payload.get("context", "")))
            if not crop:
                return AgentResult("किस फसल की जानकारी चाहिए? कृपया फसल का नाम लिखें।", "needs_input",
                                   metadata={"topic": "clarification"})
            answer, refs = build_crop_production_followup(question, crop_hint=crop, reasoning_generator=None)
            if answer and refs:
                return AgentResult(answer, references=refs,
                                   metadata={"topic": "crop_guide_followup", "crop": crop})
            # Carry the resolved crop into research when the guide has no section.
            return AgentResult("", "unavailable", metadata={"topic": "crop_guide_followup",
                               "crop": crop, "research_question": f"{crop}: {question}"})
        text = f"{payload['context']} किसान का प्रश्न: {payload['question']}" if payload.get("context") else payload["question"]
        result = self.advisor._answer_legacy(text)
        return AgentResult(result["answer"], status=result.get("status", "ok"), references=result.get("references", []),
                           metadata={"topic": result.get("topic", "rag"),
                                     **({"research_offer": result["research_offer"]} if result.get("research_offer") else {})})
