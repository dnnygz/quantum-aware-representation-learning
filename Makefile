PYTHON ?= python
CONFIG ?= configs/data.yaml

.PHONY: data metadata cohort audit-er compare-cohorts preprocess phase1c-protocol baseline-cv baseline-final audit-phase1c-semantics test clean-generated

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

preprocess:
	$(PYTHON) -m src.data.preprocessing --config $(CONFIG)

phase1c-protocol:
	$(PYTHON) -m src.data.phase1c_protocol --config $(CONFIG)

baseline-cv:
	$(PYTHON) -m src.data.classical_baseline --config $(CONFIG) --stage cv

baseline-final:
	$(PYTHON) -m src.data.classical_baseline --config $(CONFIG) --stage final

audit-phase1c-semantics:
	$(PYTHON) -m src.data.phase1c_semantic_audit --config $(CONFIG)

test:
	$(PYTHON) -m pytest

clean-generated:
	@echo "Generated data are intentionally not deleted automatically; raw snapshots are immutable."
