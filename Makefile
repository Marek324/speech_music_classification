# Environment sync

ENV_NAME = bp_env

freeze:
	pip freeze > requirements.txt
	mamba env export > env.yml

update:
	mamba env update --name $(ENV_NAME) --file env.yml

setup:
	mamba env create --file env.yml

.PHONY: freeze update setup activate
