.PHONY: install scrape process train

install:
	poetry install
	poetry run pre-commit install
	poetry run ipython kernel install --user

test:
	poetry run pre-commit run -a

scrape:
	$(eval date := $(shell date -u +"%Y-%m-%d"))
	$(eval file_name := data/raw/$(date).json)
	rm -f $(file_name)
	poetry run scrapy runspider src/scripts/scrape.py -o $(file_name) --loglevel=INFO

process:
	poetry run python src/scripts/process.py

train:
	poetry run python src/scripts/train.py

recommend:
	poetry run python src/scripts/recommend.py
