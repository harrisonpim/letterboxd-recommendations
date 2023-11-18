from pathlib import Path
import torch
import pandas as pd
import typer
from scrapy.crawler import CrawlerProcess
from src.scripts.scrape import LetterboxdSpider
from tempfile import NamedTemporaryFile
from src.model import Recommender
from rich import console

console = console.Console(highlight=False)
app = typer.Typer()

model_dir = Path("data/models")
model_path = sorted(model_dir.glob("*.pkl"))[-1]
console.print(f"Loading model from {model_path}")
model = Recommender.load(model_path)


@app.command(help="Recommend movies based on your Letterboxd ratings")
def main(
    username: str = typer.Option(..., prompt=True, help="Your Letterboxd username")
):
    with console.status("Scraping Letterboxd..."):
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

    console.print(f"Found {len(data)} ratings for {username}")

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

    ratings = model.predict(user_embedding=user_embedding)

    console.print(ratings.sort_values("predicted-rating", ascending=False).head(10))


if __name__ == "__main__":
    app()
