# Environment sync

ENV_NAME = bp_env
REQ_FILE = .requirements.txt

setup:
	mamba create -n $(ENV_NAME) -f $(REQ_FILE) -y

freeze:
	pip list --format freeze > $(REQ_FILE)

activate:
	@echo "To activate environment run:\nmamba activate $(ENV_NAME)"

.PHONY: freeze setup activate
