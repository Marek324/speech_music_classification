# Environment sync

ENV_NAME = bp_env
REQ_FILE = .requirements.txt

METACENTER_MAMBA_CHANNEL1 = https://s3.cl5.du.cesnet.cz/f463c41e_8ad6_4e8e_a0ad_4b3370c11a04:conda-forge-s3chnl
METACENTER_MAMBA_CHANNEL2 = https://s3.cl5.du.cesnet.cz/f463c41e_8ad6_4e8e_a0ad_4b3370c11a04:bioconda-s3chnl

setup:
	mamba create -n $(ENV_NAME) -f $(REQ_FILE) -y -c $(METACENTER_MAMBA_CHANNEL1) -c $(METACENTER_MAMBA_CHANNEL2) --override-channels

freeze:
	pip list --format freeze --not-required > $(REQ_FILE)

activate:
	@echo "To activate environment run:\nmamba activate $(ENV_NAME)"

dataset:
	git clone git@hf.co:$(DATASET_URL) dataset

.PHONY: freeze setup activate env dataset
