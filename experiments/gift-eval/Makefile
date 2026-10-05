.PHONY: download-gift-eval-data upload-data-to-s3 sync-ci-results sync-results sync-replication verify-all replication-table verify-replication-pilot verify-replication-full

RUN_ID ?=

download-gift-eval-data:
	@hf download Salesforce/GiftEval --repo-type=dataset --local-dir=./data/gift-eval

upload-data-to-s3: download-gift-eval-data
	@aws s3 sync ./data/gift-eval s3://foundationforecast-gift-eval/data/gift-eval

sync-ci-results:
	@mkdir -p ./results/ci
	@aws s3 sync s3://foundationforecast-gift-eval/results/ci ./results/ci

sync-results:
	@mkdir -p ./results
	@aws s3 sync s3://foundationforecast-gift-eval/results ./results

sync-replication:
	@test -n "$(RUN_ID)" || (echo "Set RUN_ID=..." && exit 1)
	@mkdir -p ./results/replication/$(RUN_ID)
	@aws s3 sync s3://foundationforecast-gift-eval/results/replication/$(RUN_ID) ./results/replication/$(RUN_ID)

verify-replication-pilot: sync-replication
	@uv run python -m src.runners.run_verify --replication --run-id $(RUN_ID) --replication-pilot

verify-replication-full: sync-replication
	@uv run python -m src.runners.run_verify --replication --run-id $(RUN_ID) --all --require-complete

verify-all: sync-results
	@uv run python -m src.runners.run_verify --all

replication-table: sync-results
	@uv run python -m src.runners.run_verify --all --verify-only \
		--table-output ./results/replication_table.csv
