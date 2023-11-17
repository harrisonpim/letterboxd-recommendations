.PHONY: install scrape process train

install:
	poetry install
	poetry run pre-commit install
	poetry run ipython kernel install --user

scrape:
	$(eval date := $(shell date -u +"%Y-%m-%d"))
	$(eval file_name := data/raw/$(date).json)
	rm -f $(file_name)
	poetry run scrapy runspider scripts/scrape.py -o $(file_name) --loglevel=INFO

process:
	poetry run python scripts/process.py

train:
	poetry run python scripts/train.py

test:
	poetry run pre-commit run -a
