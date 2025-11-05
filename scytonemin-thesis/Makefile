.PHONY: setup quickstart reproduce figures audit docs clean

PYTHON ?= python
VENV := .venv
ACTIVATE := source $(VENV)/bin/activate

setup:
	@test -d $(VENV) || ($(PYTHON) -m venv $(VENV) && $(ACTIVATE) && pip install --upgrade pip)
	@$(ACTIVATE) && pip install -r env/requirements.txt

quickstart:
	@$(ACTIVATE) && pytest tests -m "quick" --maxfail=1
	@$(ACTIVATE) && python notebooks/run_samples.py

reproduce:
	@$(ACTIVATE) && python notebooks/run_full_pipeline.py

figures:
	@$(ACTIVATE) && python notebooks/render_figures.py

audit:
	@$(ACTIVATE) && pytest tests --maxfail=1

docs:
	@$(ACTIVATE) && mkdocs build -q

clean:
	rm -rf $(VENV) .pytest_cache .mypy_cache site
