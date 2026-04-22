create-venv:
	~/.local/bin/virtualenv venv

install:
	pip install -r requirements.txt

separate:
	@python data_processing/separate.py $(filter-out $@,$(MAKECMDGOALS))

compare:
	@python training/compare.py $(filter-out $@,$(MAKECMDGOALS))
	
train:
	@python training/train.py $(ARGS)

predict:
	@python prediction/predict.py $(filter-out $@,$(MAKECMDGOALS))

%:
	@
.PHONY: all separate compare train predict
