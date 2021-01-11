PIP				:= pip
PYTHON			:= python
REQUIREMENTS	:= requirements.txt
SETUP_PY		:= setup.py

.PHONY: all dep install


all: dep install


dep: $(REQUIREMENTS)
	$(PIP) install -r $^


install: dep $(SETUP_PY)
	$(PYTHON) $(SETUP_PY) install
