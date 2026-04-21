PYTHON ?= python

install:
	$(PYTHON) -m pip install --upgrade pip
	pip install -r requirements.txt

train:
	$(PYTHON) -m src.train

simulate:
	$(PYTHON) -m src.simulate_production

api:
	$(PYTHON) -m uvicorn api.main:app --reload

dashboard:
	$(PYTHON) -m streamlit run app/streamlit_app.py

test:
	pytest -q
