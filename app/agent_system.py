"""Bounded, in-process multi-agent orchestration with typed task messages.

Agents exchange observations, never executable instructions. The audit trace stores
plans, tool outcomes and brief decision summaries, not private chain-of-thought.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Callable
from uuid import uuid4


@dataclass
class AgentResult:
    answer: str
    status: str = "ok"
    references: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentMessage:
    id: str
    sender: str
    recipient: str
    kind: str
    goal: str
    payload: dict
    reply_to: str | None = None


class MessageBus:
    def __init__(self, agents: dict, max_calls: int = 8):
        self.agents = agents
        self.max_calls = max_calls
        self.messages: list[AgentMessage] = []
        self.active: list[str] = []
        self.calls = 0
        self.cache: dict[str, AgentResult] = {}

    def ask(self, sender: str, recipient: str, goal: str, payload: dict) -> AgentResult:
        if recipient not in self.agents:
            raise ValueError(f"Unknown agent: {recipient}")
        request = AgentMessage(uuid4().hex, sender, recipient, "request", goal, payload)
        self.messages.append(request)
        cache_key = json.dumps([recipient, goal, payload], sort_keys=True, ensure_ascii=False)
        if cache_key in self.cache:
            result = self.cache[cache_key]
        elif self.calls >= self.max_calls or recipient in self.active:
            result = AgentResult("कार्य सीमा पूरी हुई; कृपया सवाल को छोटा करें।", "unavailable")
        else:
            self.calls += 1
            self.active.append(recipient)
            try:
                result = self.agents[recipient].run(goal, payload, self)
                if not isinstance(result, AgentResult):
                    raise TypeError("Agent must return AgentResult")
            except Exception as exc:
                # Do not expose credentials or upstream response bodies in chat.
                result = AgentResult("इस हिस्से का डेटा अभी उपलब्ध नहीं है।", "unavailable",
                                     evidence={"error_type": type(exc).__name__})
            finally:
                self.active.pop()
            self.cache[cache_key] = result
        self.messages.append(AgentMessage(uuid4().hex, recipient, sender, "result", goal,
                                          asdict(result), reply_to=request.id))
        return result


class ToolAgent:
    def __init__(self, handler: Callable):
        self.handler = handler

    def run(self, goal, payload, bus):
        return self.handler(goal, payload)


class PriceAgent:
    def __init__(self, handler: Callable):
        self.handler = handler

    def run(self, goal, payload, bus):
        freshness = bus.ask("prices", "market_data", "Check and refresh official market records", payload)
        result = self.handler(goal, payload)
        result.evidence["refresh"] = freshness.evidence
        return result


class PesticideAgent:
    def __init__(self, handler: Callable):
        self.handler = handler

    def run(self, goal, payload, bus):
        result = self.handler(goal, payload)
        if re.search(r"spray|छिड़क|बारिश|rain|weather|मौसम", payload["question"], re.I):
            weather = bus.ask("pesticides", "weather", "Check weather before spray advice", payload)
            result.evidence["weather_check"] = asdict(weather)
            result.references = list(dict.fromkeys(result.references + weather.references))
            result.answer += "\n\nछिड़काव से पहले मौसम:\n" + weather.answer
            if weather.status != "ok":
                result.status = "partial" if result.status == "ok" else result.status
                result.answer += "\nमौसम की पुष्टि नहीं हुई है; अभी छिड़काव का समय तय नहीं किया जा सकता।"
        return result


class Coordinator:
    def __init__(self, agents: dict, planner: Callable | None = None):
        self.agents = agents
        self.planner = planner

    @staticmethod
    def domains(question: str) -> list[str]:
        patterns = {
            "weather": r"\b(weather|mausam|mosam|rain|barish|temperature)\b|मौसम|बारिश|तापमान|वर्षा",
            "prices": r"\b(price|prices|rate|mandi|bhav|bhaav|daam|msp)\b|भाव|कीमत|दाम|मंडी|समर्थन मूल्य",
            "pesticides": r"\b(pesticide|pesticides|insecticide|fungicide|herbicide|spray|dawai|keet|rog|disease|pest)\b|दवा|दवाई|कीट|रोग|छिड़क|फफूंद|लक्षण",
        }
        selected = [name for name, pattern in patterns.items() if re.search(pattern, question, re.I)]
        if not selected and re.search(r"\b(kheti|cultivation)\b|खेती|how to (grow|cultivate)", question, re.I):
            selected.append("agronomy")
        return selected

    def plan(self, question: str) -> tuple[list[str], str]:
        required = self.domains(question)
        selected, source = required or ["agronomy"], "rules"
        # Straightforward tool queries do not need an LLM round trip.
        if self.planner and (not required or len(required) > 1):
            try:
                raw = self.planner(
                    'Return only JSON {"agents": [names]}. Select specialists needed for this farmer question. '
                    'Allowed names: weather, prices, pesticides, agronomy. '
                    'Use agronomy for crop planning or general agriculture. Do not answer the question.\n'
                    + json.dumps({"question": question}, ensure_ascii=False)
                )
                proposed = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())["agents"]
                if not isinstance(proposed, list) or not proposed or len(proposed) > 4:
                    raise ValueError("Invalid plan")
                if any(not isinstance(n, str) or n not in {"weather", "prices", "pesticides", "agronomy"} for n in proposed):
                    raise ValueError("Unknown specialist")
                selected, source = list(dict.fromkeys(required + proposed)), "model"
            except Exception:
                source = "rules_fallback"
        # Pesticide agent obtains the weather evidence itself for spray questions.
        if "pesticides" in selected and "weather" in selected and re.search(r"spray|छिड़क", question, re.I):
            selected.remove("weather")
        return selected, source

    def answer(self, question: str, context: str = "") -> dict:
        goal = f"Answer the farmer's question using dated evidence: {question}"
        names, source = self.plan(question)
        bus = MessageBus(self.agents)
        payload = {"question": question, "context": context}
        results = []
        decisions = []
        for name in names:
            result = bus.ask("coordinator", name, goal, payload)
            results.append((name, result))
            decisions.append({"agent": name, "status": result.status,
                              "decision": "Use cited tool evidence" if result.status == "ok" else "Report missing or stale evidence; do not invent values"})
        refs = list(dict.fromkeys(ref for _, r in results for ref in r.references))
        text = "\n\n".join(r.answer for _, r in results)
        statuses = [r.status for _, r in results]
        status = "ok" if all(s == "ok" for s in statuses) else "partial" if "ok" in statuses else statuses[0]
        metadata = results[0][1].metadata if len(results) == 1 else {}
        topic = {"prices": "price", "pesticides": "pesticide", "weather": "weather", "agronomy": "rag"}.get(names[0]) if len(names) == 1 else "multi_agent"
        return {"answer": text, "references": refs, "retrieved": [], "topic": topic,
                **metadata, "agent_status": status,
                "agent_trace": {"goal": goal, "plan": names, "planner": source,
                                "decisions": decisions, "messages": [asdict(m) for m in bus.messages],
                                "calls": bus.calls}}


def build_coordinator(advisor):
    from app.agent_tools import AdvisorTools
    handlers = AdvisorTools(advisor)

    def plan_with_model(prompt):
        if os.getenv("KISAANAI_LLM_PLANNER", "1").lower() in {"0", "false", "no"}:
            raise RuntimeError("Model planning disabled")
        generator = advisor._get_generator_for_model(advisor.cfg.complex_generator_model or advisor.cfg.generator_model)
        if generator is None:
            raise RuntimeError("Planner model unavailable")
        return generator.generate(prompt)

    return Coordinator({"weather": ToolAgent(handlers.weather),
                        "market_data": ToolAgent(handlers.refresh_market),
                        "prices": PriceAgent(handlers.prices),
                        "pesticides": PesticideAgent(handlers.pesticides),
                        "agronomy": ToolAgent(handlers.agronomy)}, planner=plan_with_model)
