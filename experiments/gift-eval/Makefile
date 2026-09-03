.PHONY: download-gift-eval-data upload-data-to-s3 sync-ci-results sync-results verify-all replication-table

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

verify-all: sync-results
	@uv run python -m src.runners.run_verify --all

replication-table: sync-results
	@uv run python -m src.runners.run_verify --all --verify-only \
		--table-output ./results/replication_table.csv
