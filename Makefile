PYTHON ?= python
MEMCAP ?=
# Small BLAS thread pools: the matrices are small and the machine may be shared.
export OMP_NUM_THREADS ?= 2
export OPENBLAS_NUM_THREADS ?= 2
export MKL_NUM_THREADS ?= 2
export VECLIB_MAXIMUM_THREADS ?= 2

.PHONY: install install-locked data run test lint clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

# Exact package versions that produced the committed reports/ (see requirements.lock).
install-locked:
	$(PYTHON) -m pip install -r requirements.lock
	$(PYTHON) -m pip install -e . --no-deps

# Download MovieLens 100K from GroupLens into data/ (cached; the licence forbids redistribution).
data:
	$(PYTHON) -c "from recsys.data import download_ml100k; download_ml100k('data')"

# Full reproduction: tuning, test evaluation, protocol comparison -> reports/metrics.json + figures.
run: data
	$(MEMCAP) $(PYTHON) -m recsys.cli run --data-dir data --out reports

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
