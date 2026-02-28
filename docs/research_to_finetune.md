# Research-Paper to Fine-Tune Workflow

## Goal
Use agriculture research papers and local historical results to improve an SLM for Western UP recommendations.

## Steps
1. Collect papers
- Agronomy, crop economics, irrigation, soil health, pest management.
- Prefer India-specific and Indo-Gangetic plain studies.

2. Extract useful supervision
- Convert findings into instruction-response format.
- Example:
  - Instruction: "Low irrigation, Rabi season, Meerut district, medium budget."
  - Response: "Mustard/Wheat strategy with expected cost and risk."

3. Merge with historical advisories
- Include past recommendations and measured outcomes.
- Keep fields: district, season, soil type, budget band, crop, yield, realized price.

4. Fine-tune with LoRA
- Keep base SLM unchanged.
- Train adapter on domain prompts.
- Evaluate before replacing production adapter.

5. Evaluation checklist
- Factuality against source context.
- Budget realism.
- Regional relevance.
- Hallucination rate.
- Safety and uncertainty behavior.
