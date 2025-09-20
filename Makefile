# Environment sync

ENV_NAME = bp_env
REQ_FILE = .requirements.txt

setup:
	mamba create -n $(ENV_NAME) -f $(REQ_FILE) -y

freeze:
	pip list --format freeze > $(REQ_FILE)

activate:
	@echo "To activate environment run:\nmamba activate $(ENV_NAME)"

env:
	@set -a
	@bash -c "source .env"
	@set +a
	@echo "Exported .env"

dataset: env
	git clone git@hf.co:$(DATASET_URL) dataset

.PHONY: freeze setup activate env dataset
