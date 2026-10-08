PYTHON ?= python
CONFIG ?= configs/data.yaml

.PHONY: data metadata cohort audit-er compare-cohorts test clean-generated

data:
	$(PYTHON) -m src.data.download --config $(CONFIG)
	$(PYTHON) -m src.data.cohort --config $(CONFIG)

metadata:
	$(PYTHON) -m src.data.download --config $(CONFIG) --metadata-only

cohort:
	$(PYTHON) -m src.data.cohort --config $(CONFIG)

audit-er:
	$(PYTHON) -m src.data.er_source_audit --config $(CONFIG)

compare-cohorts:
	$(PYTHON) -m src.data.compare_cohorts --config $(CONFIG)

test:
	$(PYTHON) -m pytest

clean-generated:
	@echo "Generated data are intentionally not deleted automatically; raw snapshots are immutable."
