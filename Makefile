SHELL := /bin/zsh
PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: help venv install init seed ingest index bootstrap run compile clean docker-build docker-up docker-down ci

help:
	@echo "Targets:"
	@echo "  make venv        - create virtualenv"
	@echo "  make install     - install dependencies"
	@echo "  make init        - initialize SQLite schema"
	@echo "  make seed        - load seed economics/advisories"
	@echo "  make ingest      - ingest raw docs"
	@echo "  make index       - build vector index"
	@echo "  make bootstrap   - init + seed + ingest + index"
	@echo "  make run         - run Streamlit app"
	@echo "  make compile     - syntax check"
	@echo "  make docker-up   - run app with Docker Compose"
	@echo "  make ci          - local CI checks"

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

init:
	$(PY) scripts/init_db.py

seed:
	$(PY) scripts/load_seed_data.py

ingest:
	$(PY) scripts/ingest_documents.py --input_dir data/raw

index:
	$(PY) scripts/build_index.py

bootstrap:
	$(PY) scripts/bootstrap_data.py

run:
	$(VENV)/bin/streamlit run streamlit_app.py

compile:
	$(PYTHON) -m compileall app scripts streamlit_app.py

clean:
	rm -rf __pycache__ app/__pycache__ scripts/__pycache__ .pytest_cache

docker-build:
	docker compose build

docker-up:
	docker compose up --build

docker-down:
	docker compose down

ci: compile
