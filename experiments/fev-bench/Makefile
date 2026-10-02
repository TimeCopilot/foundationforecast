.PHONY: sync-ci-results sync-results

sync-ci-results:
	@mkdir -p ./results/ci
	@aws s3 sync s3://foundationforecast-fev-bench/results/ci ./results/ci

sync-results:
	@mkdir -p ./results
	@aws s3 sync s3://foundationforecast-fev-bench/results ./results
