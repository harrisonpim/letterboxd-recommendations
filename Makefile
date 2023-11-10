.PHONY: install 

install:
	poetry install
	poetry run pre-commit install
	poetry run ipython kernel install --user
