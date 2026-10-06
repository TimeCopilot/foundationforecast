# Gift-Eval experiment — common tasks (run from experiments/gift-eval/).
#
# Modal runs need MODAL_TOKEN_* , HF_TOKEN , and AWS creds in the environment
# (same as GitHub Actions secrets). Long grids: DETACHED=1 make run-replication-full ...

.DEFAULT_GOAL := help

.PHONY: help
.PHONY: download-gift-eval-data upload-data-to-s3
.PHONY: run-ci sync-ci-results verify-ci
.PHONY: run-replication-pilot run-replication-full
.PHONY: sync-replication verify-replication-pilot verify-replication-full
.PHONY: sync-results verify-all replication-table

RUN_ID ?=
DETACHED ?=
FORCE ?=

# Pass -d to modal run for detached (background) execution.
MODAL_RUN = uv run modal run $(if $(filter 1 true yes,$(DETACHED)),-d,)

## help — list targets (default)
help:
	@echo "Gift-Eval Makefile"
	@echo ""
	@echo "Data"
	@echo "  download-gift-eval-data   HF dataset → ./data/gift-eval"
	@echo "  upload-data-to-s3         sync local dataset to S3"
	@echo ""
	@echo "CI (configs/ci_subset.yaml — strict per-job HF verify in pytest)"
	@echo "  run-ci                    Modal GPU eval + overwrite results/ci on S3"
	@echo "  sync-ci-results           S3 → ./results/ci"
	@echo "  verify-ci                 sync + pytest tests/test_replication.py"
	@echo ""
	@echo "Replication (configs/replication/ — pilot strict, full aggregate verify)"
	@echo "  run-replication-pilot     RUN_ID=...  — 7× m4_weekly/short on Modal"
	@echo "  run-replication-full      RUN_ID=...  — 7×97 grid (skips existing unless FORCE=1)"
	@echo "  sync-replication          RUN_ID=...  — S3 → ./results/replication/\$$RUN_ID"
	@echo "  verify-replication-pilot  RUN_ID=...  — per-job MASE/CRPS vs HF"
	@echo "  verify-replication-full   RUN_ID=...  — leaderboard aggregate + require complete"
	@echo ""
	@echo "Full benchmark (configs/models.yaml — production results/ on S3)"
	@echo "  sync-results              S3 → ./results"
	@echo "  verify-all                strict HF verify for all reference models"
	@echo "  replication-table         analysis CSV (verify-only, no assert)"
	@echo ""
	@echo "Options: DETACHED=1 for Modal -d ; FORCE=1 to re-run all replication jobs"

# --- Data -------------------------------------------------------------------

download-gift-eval-data:
	@hf download Salesforce/GiftEval --repo-type=dataset --local-dir=./data/gift-eval

upload-data-to-s3: download-gift-eval-data
	@aws s3 sync ./data/gift-eval s3://foundationforecast-gift-eval/data/gift-eval

# --- CI ---------------------------------------------------------------------
# GitHub Actions: make run-ci && make verify-ci

run-ci:
	@$(MODAL_RUN) -m src.runners.run_modal::run_ci

sync-ci-results:
	@mkdir -p ./results/ci
	@aws s3 sync s3://foundationforecast-gift-eval/results/ci ./results/ci

verify-ci: sync-ci-results
	@uv run pytest tests/test_replication.py -n 0 -x

# --- Replication ------------------------------------------------------------

run-replication-pilot:
	@test -n "$(RUN_ID)" || (echo "Set RUN_ID=..." && exit 1)
	@$(MODAL_RUN) -m src.runners.run_modal::run_replication_pilot --run-id $(RUN_ID)

run-replication-full:
	@test -n "$(RUN_ID)" || (echo "Set RUN_ID=..." && exit 1)
	@$(MODAL_RUN) -m src.runners.run_modal::run_replication_full --run-id $(RUN_ID) \
		$(if $(filter 1 true yes,$(FORCE)),--force,)

sync-replication:
	@test -n "$(RUN_ID)" || (echo "Set RUN_ID=..." && exit 1)
	@mkdir -p ./results/replication/$(RUN_ID)
	@aws s3 sync s3://foundationforecast-gift-eval/results/replication/$(RUN_ID) ./results/replication/$(RUN_ID)

verify-replication-pilot: sync-replication
	@uv run python -m src.runners.run_verify --replication --run-id $(RUN_ID) --replication-pilot

verify-replication-full: sync-replication
	@uv run python -m src.runners.run_verify --replication --run-id $(RUN_ID) --all --require-complete

# --- Full benchmark ---------------------------------------------------------

sync-results:
	@mkdir -p ./results
	@aws s3 sync s3://foundationforecast-gift-eval/results ./results

verify-all: sync-results
	@uv run python -m src.runners.run_verify --all

replication-table: sync-results
	@uv run python -m src.runners.run_verify --all --verify-only \
		--table-output ./results/replication_table.csv
