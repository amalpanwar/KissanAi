# Architecture - KisaanAI (Western UP)

## Objective
Build a local-language, region-specific agriculture AI system for Western Uttar Pradesh that can:
- Recommend profitable crops based on budget and conditions
- Provide condition-aware yield guidance
- Store historical advisories and outcomes
- Improve using RAG + SLM fine-tuning

## System Design
1. Data Layer
- Inputs: research papers, advisories, district economics, outcomes.
- Storage:
  - SQLite for structured historical records.
  - Numpy vector index for semantic retrieval.

2. Knowledge Processing
- Document parsing from TXT/CSV/JSON.
- Chunking with overlap for context continuity.
- Multilingual embeddings for Hindi/English retrieval.

3. Retrieval Layer (RAG)
- Query embedding.
- Cosine similarity top-k search over vector index.
- Retrieved chunks passed to prompt template.

4. Generation Layer (SLM)
- Base model: TinyLlama 1.1B chat (replaceable).
- Hindi-first instruction prompt.
- Output format enforces crop, budget, conditions, and risk.

5. Learning Loop
- Advisory + outcome pairs become supervised data.
- LoRA fine-tuning on domain-specific instructions.
- Rebuild model adapters periodically (monthly/quarterly).

## Historical Data Schema
- `research_documents`
- `advisories`
- `outcomes`
- `crop_economics`

## Deployment Path
1. Pilot in 2-3 districts (Meerut, Muzaffarnagar, Baghpat).
2. Validate with KVK/agri experts.
3. Add weather and mandi APIs.
4. Move from Numpy index to FAISS/Milvus for scale.

## Guardrails
- Surface confidence and uncertainty.
- Do not provide pesticide dosage without verified source context.
- Log all recommendations for audit and model improvement.
