# Convenience targets. `make setup` prepares everything; `make run` opens the app.
.PHONY: install data setup run test

install:        ## Create the venv and install dependencies
	uv sync

data:           ## Fetch boundaries + Landsat heat series, build graph + index (cached)
	uv run python scripts/setup_data.py

setup: install data  ## Full setup from a clean checkout

run:            ## Launch the dashboard
	uv run streamlit run app/dashboard.py

test:
	uv run pytest -q
