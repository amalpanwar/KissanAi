# KisaanAI - Western Uttar Pradesh Agriculture RAG System

KisaanAI is a local-language (Hindi-first) AI assistant for Western Uttar Pradesh farmers.
It combines:
- RAG over agriculture research + regional knowledge
- Historical data storage for advisories and outcomes
- Small Language Model (SLM) fine-tuning pipeline
- Crop recommendation by budget, land, season, and conditions

## Target Region (Phase 1)
- Western Uttar Pradesh districts (e.g., Meerut, Muzaffarnagar, Baghpat, Saharanpur, Shamli, Bulandshahr)

## Core Capabilities
- Suggest profitable crops by season and district
- Explain ideal conditions (soil, temperature, rainfall, irrigation)
- Budget-aware planning (input cost ranges and expected returns)
- Track historical advisories and outcomes for iterative improvement
- Answer in local Hindi register while preserving technical accuracy

## Project Structure
- `app/` - core Python modules
- `data/` - raw and processed documents/datasets
- `scripts/` - ingestion, indexing, and training entrypoints
- `configs/` - model and pipeline configs
- `docs/` - architecture and operating guide

## Quick Start
1. Create environment
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Add source documents in `data/raw/`:
   - Research papers (PDF/TXT)
   - Government advisories
   - Historical crop/yield/cost files (CSV)

3. Build database and vector index
   ```bash
   python scripts/init_db.py
   python scripts/ingest_documents.py --input_dir data/raw
   python scripts/build_index.py
   ```

4. Query the assistant
   ```bash
   python scripts/query_cli.py --q "मेरे पास 2 एकड़ जमीन है, गेहूं या आलू में कौन बेहतर रहेगा?"
   ```

5. Fine-tune SLM (LoRA)
   ```bash
   python scripts/prepare_finetune_data.py
   python scripts/finetune_slm.py
   ```

## Makefile Commands
Use these shortcuts:
```bash
make install
make bootstrap
make run
```

Other useful targets:
```bash
make compile
make docker-up
make docker-down
make ci
```

## Streamlit Hosting (Local)
1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
2. Initialize and index data
   ```bash
   python scripts/init_db.py
   python scripts/load_seed_data.py
   python scripts/ingest_documents.py --input_dir data/raw
   python scripts/build_index.py
   ```
3. Run Streamlit app
   ```bash
   streamlit run streamlit_app.py
   ```
4. Open local URL shown by Streamlit (usually `http://localhost:8501`)

## Docker Hosting
1. Build and start
   ```bash
   docker compose up --build
   ```
2. Open:
   - `http://localhost:8501`

Notes:
- Container startup runs `scripts/bootstrap_data.py` to initialize DB/index.
- Bootstrapping is idempotent (safe on restart). Use force rebuild manually:
  ```bash
  docker compose run --rm kisaanai-streamlit python scripts/bootstrap_data.py --force
  ```

## Streamlit Community Cloud
1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create app with:
   - Repository: your GitHub repo
   - Branch: main (or your deployment branch)
   - Main file path: `streamlit_app.py`
3. Ensure files are committed:
   - `requirements.txt`
   - `.streamlit/config.toml`
   - `configs/pipeline.yaml`
4. Deploy.

Notes:
- First startup can take time due to model download.
- For faster cold starts, switch to a smaller embedding/generation model in `configs/pipeline.yaml`.

## GitHub Actions CI
- Workflow file: `.github/workflows/ci.yml`
- Runs on push and pull requests.
- Checks:
  - Python syntax compile for `app/`, `scripts/`, `streamlit_app.py`
  - Required deployment files exist

## Push To Your GitHub (`amalpanwar`)
You need to create an actual repository under your account, then push this code to it.

Example:
```bash
git init
git add .
git commit -m "Initial KisaanAI RAG + Streamlit setup"
git branch -M main
git remote add origin https://github.com/amalpanwar/<your-repo-name>.git
git push -u origin main
```

If using GitHub CLI:
```bash
gh auth login
gh repo create <your-repo-name> --public --source=. --remote=origin --push
```

## Data.gov.in + 15-Day LSTM Forecast
Use these scripts to fetch commodity data from Data.gov.in and forecast 15 days ahead.

1. One-time setup (safe)
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and set:
   - `DATA_GOV_API_KEY`
   - `DATA_GOV_RESOURCE_ID` (default already set)
   - optional default filters (`DATA_GOV_STATE`, `DATA_GOV_DISTRICT`, `DATA_GOV_COMMODITY`)

2. Fetch dataset records (no repeated API/resource args)
   ```bash
   python scripts/fetch_datagov_commodity.py
   ```

3. Train LSTM and predict 15 days
   ```bash
   python scripts/lstm_forecast_15d.py --horizon 15
   ```

Notes:
- Override defaults any time with CLI flags (e.g. `--commodity`, `--district`).
- If date/value column names differ, pass `--date_col` and `--value_col`.
- Default LSTM settings: `lookback=30`, `epochs=120`.
- By default, script auto-broadens scope (`district -> state -> commodity -> all`) if points are too few for stable training (`--min_points`, default 90).
- In Streamlit sidebar, use `Refresh Market Data` to fetch fresh rows; forecast uses latest CSV immediately.

## Notes
- This scaffold is production-oriented but intentionally lightweight.
- Replace sample datasets with verified district-level data for deployment.
- Validate recommendations with agriculture experts/KVK before field rollout.
