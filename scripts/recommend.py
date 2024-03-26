from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import torch
import typer
from rich import console
from scrapy.crawler import CrawlerProcess

from scripts.scrape import LetterboxdSpider
from src.letterboxd import scrape_watchlist
from src.model import Recommender

console = console.Console(highlight=False)
app = typer.Typer()

model_dir = Path("data/models")
model_path = sorted(model_dir.glob("*.pkl"))[-1]
console.print(f"Loading model from {model_path}")
model = Recommender.load(model_path)


@app.command(help="Recommend movies based on your Letterboxd ratings")
def main(
    username: str = typer.Option(..., prompt=True, help="Your Letterboxd username"),
    from_watchlist: bool = typer.Option(
        False,
        "--from-watchlist",
        help="Recommend movies from your watchlist instead of your ratings",
    ),
    ignore_watched: bool = typer.Option(
        True,
        "--ignore-watched",
        help="Don't recommend movies you've already watched",
    ),
):
    with console.status(f"Scraping {username}'s ratings from letterboxd..."):
        temporary_file = NamedTemporaryFile()
        process = CrawlerProcess(
            settings={
                "FEEDS": {
                    temporary_file.name: {"format": "json"},
                },
            },
            install_root_handler=False,
        )
        process.crawl(LetterboxdSpider, target_user=username)
        process.start()

        data = pd.read_json(temporary_file.name, orient="records")

    console.print(f"Scraped {len(data)} ratings from {username}.")

    film_indices = []
    ratings = []
    for _, rating in data.iterrows():
        if rating["film-slug"] in model.dataset.film_to_index:
            film_indices.append(model.dataset.film_to_index[rating["film-slug"]])
            ratings.append(rating["rating"])

    # prepare the data for the model
    user_embedding = model.get_user_embedding_from_ratings(
        film_indices=torch.tensor(film_indices),
        ratings=torch.tensor(ratings).unsqueeze(1),
    )

    console.print(f"Generated an embedding based on {username}'s ratings...")

    film_slugs_to_predict_against = model.dataset.film_to_index
    if from_watchlist:
        user_watchlist = scrape_watchlist(username)
        film_slugs_to_predict_against = [
            slug for slug in user_watchlist if slug in model.dataset.film_to_index
        ]
    if ignore_watched:
        user_watched = set(data[data["rating"].notnull()]["film-slug"].unique())
        film_slugs_to_predict_against = [
            slug for slug in film_slugs_to_predict_against if slug not in user_watched
        ]

    film_indices_to_predict_against = torch.Tensor(
        [model.dataset.film_to_index[slug] for slug in film_slugs_to_predict_against]
    ).int()

    ratings = model.predict(
        user_embedding=user_embedding, film_indices=film_indices_to_predict_against
    )

    console.print(ratings.sort_values("predicted-rating", ascending=False).head(30))


if __name__ == "__main__":
    app()
