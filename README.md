# Letterboxd recommendations

I want to know what's good.

## Usage

This project is written in python, and orchestrated with [poetry](https://python-poetry.org/) and a [Makefile](https://www.gnu.org/software/make/).

To get started, run `make install` to install the dependencies.

- `make scrape` will scrape the letterboxd website for ratings from n users (default 1000)
- `make process` will augment the raw data into a more usable format
- `make train` will train a model on the data
