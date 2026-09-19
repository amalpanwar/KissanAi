import unittest
from app.agent_system import AgentResult, Coordinator, MessageBus, PesticideAgent, PriceAgent, ToolAgent


def handler(answer, status="ok"):
    return ToolAgent(lambda goal, payload: AgentResult(answer, status, ["official source"]))


class AgentTests(unittest.TestCase):
    def agents(self):
        return {"weather": handler("Weather evidence"), "market_data": handler("Fresh data"),
                "prices": PriceAgent(lambda g, p: AgentResult("Price evidence")),
                "pesticides": PesticideAgent(lambda g, p: AgentResult("Label evidence")),
                "agronomy": handler("Crop evidence")}

    def test_mixed_query_runs_both_specialists(self):
        result = Coordinator(self.agents()).answer("Meerut weather and wheat price")
        self.assertEqual(result["agent_trace"]["plan"], ["weather", "prices"])
        self.assertIn("Weather evidence", result["answer"])
        self.assertIn("Price evidence", result["answer"])
        requests = [(m["sender"], m["recipient"]) for m in result["agent_trace"]["messages"] if m["kind"] == "request"]
        self.assertIn(("prices", "market_data"), requests)

    def test_spray_agent_contacts_weather(self):
        result = Coordinator(self.agents()).answer("गेहूं में दवा छिड़काव और मौसम बताओ")
        self.assertEqual(result["agent_trace"]["plan"], ["pesticides"])
        requests = [(m["sender"], m["recipient"]) for m in result["agent_trace"]["messages"] if m["kind"] == "request"]
        self.assertIn(("pesticides", "weather"), requests)
        self.assertIn("Weather evidence", result["answer"])

    def test_invalid_model_plan_falls_back(self):
        result = Coordinator(self.agents(), lambda p: '{"agents":["shell"]}').answer("weather and prices")
        self.assertEqual(result["agent_trace"]["planner"], "rules_fallback")
        self.assertEqual(result["agent_trace"]["plan"], ["weather", "prices"])

    def test_model_cannot_drop_required_domain(self):
        result = Coordinator(self.agents(), lambda p: '{"agents":["weather"]}').answer("weather and prices")
        self.assertIn("prices", result["agent_trace"]["plan"])

    def test_failure_does_not_hide_other_answer(self):
        agents = self.agents()
        agents["weather"] = handler("Weather unavailable", "unavailable")
        result = Coordinator(agents).answer("weather and price")
        self.assertEqual(result["agent_status"], "partial")
        self.assertIn("Price evidence", result["answer"])

    def test_cycle_bounded(self):
        class Cyclic:
            def run(self, g, p, bus):
                return bus.ask("loop", "loop", g, p)
        bus = MessageBus({"loop": Cyclic()})
        self.assertEqual(bus.ask("coordinator", "loop", "g", {}).status, "unavailable")
        self.assertEqual(bus.calls, 1)

    def test_no_weather_spray_timing_not_asserted(self):
        agents = self.agents()
        agents["weather"] = handler("Location missing", "needs_input")
        result = Coordinator(agents).answer("pesticide spray")
        self.assertIn("समय तय नहीं", result["answer"])

    def test_budget_enforced(self):
        bus = MessageBus(self.agents(), max_calls=0)
        self.assertEqual(bus.ask("coordinator", "weather", "g", {}).status, "unavailable")

if __name__ == "__main__":
    unittest.main()
