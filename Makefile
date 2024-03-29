.PHONY: install test scrape process train recommend api infra-start infra-stop infra-export infra-ssh

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
	poetry run scrapy runspider scripts/scrape.py -o $(file_name) --loglevel=INFO

process:
	poetry run python scripts/process.py

train:
	poetry run python scripts/train.py

recommend:
	poetry run python scripts/recommend.py

api:
	poetry run uvicorn api.main:app --reload

infra-start:
	poetry run pulumi up --cwd infra --stack letterboxd --yes

infra-stop:
	poetry run pulumi destroy --cwd infra --stack letterboxd --yes

infra-export:
	export $$(poetry run pulumi stack output --shell --cwd infra --stack letterboxd)
