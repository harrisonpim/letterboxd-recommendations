# Letterboxd Recommendations

I want to know what's good.

## Usage

This project is written in python, and orchestrated with [poetry](https://python-poetry.org/) and a [Makefile](https://www.gnu.org/software/make/).

To get started, run `make install` to install the dependencies. Then,

- `make scrape` will scrape the letterboxd website for ratings from n users (default 1000)
- `make process` will augment the raw data into a more useful format for training
- `make train` will train a model on the data
- `make recommend` will run a CLI to recommend movies for a specified user
- `make api` will start a FastAPI server to recommend movies for a specified user or find similar movies based on the learned embeddings
- `make test` will run the project's tests

Training the model requires a GPU, and the infrastructure for this is managed with pulumi. See the [infrastructure README](./infra/README.md) for more information.
