import random
import torch
from fastapi import FastAPI
from src.model import Recommender
from scripts.scrape import stars_to_rating
import httpx
from pathlib import Path
import scrapy

data_dir = Path("data/processed/2023-11-19")
model_dir = Path("data/models")
model_path = sorted(model_dir.glob("*.pkl"))[-1]
model = Recommender.load(model_path)

app = FastAPI()


@app.get("/predict")
def predict(username: str):
    first_page_url = f"https://letterboxd.com/{username}/films/by/date/"
    response = httpx.get(first_page_url)
    response.raise_for_status()
    html = scrapy.Selector(text=response.text)
    ratings = html.css("li.poster-container")
    data = []
    for rating in ratings:
        rating_string = rating.css("span.rating::text").get()
        if rating_string is None:
            continue
        data.append(
            {
                "username": username,
                "film-slug": rating.css("div.poster::attr(data-film-slug)").get(),
                "rating": stars_to_rating(rating_string),
            }
        )

    film_indices = []
    ratings = []
    for rating in data:
        if rating["film-slug"] in model.dataset.film_to_index:
            film_indices.append(model.dataset.film_to_index[rating["film-slug"]])
            ratings.append(rating["rating"])

    # prepare the data for the model
    user_embedding = model.get_user_embedding_from_ratings(
        film_indices=torch.tensor(film_indices),
        ratings=torch.tensor(ratings).unsqueeze(1),
    )

    # get the recommendations
    recommendations = model.predict(
        user_embedding, film_indices=random.sample(range(len(model.dataset.films)), 10)
    )

    return recommendations
