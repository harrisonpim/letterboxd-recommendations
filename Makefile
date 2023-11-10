.PHONY: install scrape process train

install:
	poetry install
	poetry run pre-commit install
	poetry run ipython kernel install --user

scrape:
	$(eval date := $(shell date -u +"%Y-%m-%d"))
	poetry run scrapy runspider scripts/scrape.py -o data/raw/$(date).json

process:
	poetry run python scripts/process.py
