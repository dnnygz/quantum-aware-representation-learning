PYTHON ?= python
CONFIG ?= configs/data.yaml

.PHONY: data metadata cohort test clean-generated

data:
	$(PYTHON) -m src.data.download --config $(CONFIG)
	$(PYTHON) -m src.data.cohort --config $(CONFIG)

metadata:
	$(PYTHON) -m src.data.download --config $(CONFIG) --metadata-only

cohort:
	$(PYTHON) -m src.data.cohort --config $(CONFIG)

test:
	$(PYTHON) -m pytest

clean-generated:
	@echo "Generated data are intentionally not deleted automatically; raw snapshots are immutable."
