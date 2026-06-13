from __future__ import annotations

from dataclasses import asdict, dataclass
import re


@dataclass
class QueryPlan:
    route: str
    reason: str
    complexity_score: float
    top_k: int
    cache_ttl_sec: int
    model_name: str | None = None
    allow_web_fallback: bool = True
    use_hyde: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class QueryAgent:
    def __init__(
        self,
        medium_generator_model: str,
        *,
        complex_generator_model: str | None = None,
        default_top_k: int = 3,
    ) -> None:
        self.medium_generator_model = medium_generator_model
        self.complex_generator_model = complex_generator_model or None
        self.default_top_k = max(1, int(default_top_k))

    def decide(self, question: str, context_part: str = "") -> QueryPlan:
        text = str(question or "").strip()
        normalized = text.lower()
        tokens = re.findall(r"[a-z0-9\u0900-\u097F]+", normalized)
        score = 0.0
        reasons: list[str] = []

        if len(tokens) >= 12:
            score += 0.24
            reasons.append("long_query")
        if len(tokens) >= 20:
            score += 0.16
            reasons.append("very_long_query")

        complex_markers = [
            "why",
            "क्यों",
            "reason",
            "explain",
            "समझाओ",
            "detail",
            "विस्तार",
            "strategy",
            "plan",
            "compare",
            "difference",
            "vs",
            "budget",
            "profit",
            "profitable",
            "laabh",
            "लाभ",
            "step",
            "कैसे",
            "kaise",
        ]
        if any(marker in normalized for marker in complex_markers):
            score += 0.28
            reasons.append("reasoning_markers")

        if context_part and len(context_part.split()) >= 10:
            score += 0.10
            reasons.append("rich_context")

        if normalized.count("?") + normalized.count("।") + normalized.count(",") >= 2:
            score += 0.08
            reasons.append("multi_clause")

        if any(marker in normalized for marker in ["today", "latest", "current", "आज", "अभी"]):
            score -= 0.10
            reasons.append("volatile_query")

        short_factoid = len(tokens) <= 8 and not any(marker in normalized for marker in complex_markers)
        if short_factoid:
            score -= 0.08
            reasons.append("short_factoid")

        score = max(0.0, min(score, 1.0))

        if short_factoid and score < 0.22:
            return QueryPlan(
                route="retrieval_only",
                reason=", ".join(reasons) or "short_factoid",
                complexity_score=score,
                top_k=max(5, self.default_top_k),
                cache_ttl_sec=12 * 60 * 60,
                model_name=None,
            )

        if score >= 0.60 and self.complex_generator_model:
            return QueryPlan(
                route="generator_complex",
                reason=", ".join(reasons) or "complex_query",
                complexity_score=score,
                top_k=max(5, self.default_top_k + 2),
                cache_ttl_sec=6 * 60 * 60,
                model_name=self.complex_generator_model,
            )

        return QueryPlan(
            route="generator_medium",
            reason=", ".join(reasons) or "medium_generator",
            complexity_score=score,
            top_k=max(5, self.default_top_k + 1),
            cache_ttl_sec=6 * 60 * 60,
            model_name=self.medium_generator_model,
        )
