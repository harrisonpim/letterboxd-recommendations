import numpy as np
from typing import List, Optional, Annotated
from pydantic import BaseModel
from fastapi import FastAPI, Query
from src.numpy_recommender import NumpyRecommender, cosine_similarity
from src.letterboxd import stars_to_rating
import httpx
from pathlib import Path
import scrapy

model_dir = Path("data/models/numpy/")
model_path = sorted(model_dir.glob("*.pkl"))[-1]
model = NumpyRecommender.load(model_path)

app = FastAPI()


class FilmRecommendation(BaseModel):
    slug: str
    url: str
    score: float


@app.get("/predict")
def predict(username: str, n: Optional[int] = 10) -> List[FilmRecommendation]:
    """
    Make predictions for the user based on their most recently watched films

    :param str username: The target username. Their recently watched films will be
    scraped and used to form a basis for making recommendations
    :param Optional[int] n: number of recommendations to make, defaults to 10
    :return List[FilmRecommendation]: A list of films which the user is likely to enjoy
    """
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
        if rating["film-slug"] in model.film_to_index:
            film_indices.append(model.film_to_index[rating["film-slug"]])
            ratings.append(rating["rating"])

    user_embedding = model.get_user_embedding_from_ratings(
        film_indices=np.array(film_indices), ratings=np.array(ratings)
    )
    user_embeddings = user_embedding.unsqueeze(0)
    film_embeddings = model.film_embeddings

    # get the recommendations
    predictions = model(user_embeddings, film_embeddings).squeeze(0)
    top_indices = np.argsort(predictions)[::-1][:n]
    recommendations = [
        FilmRecommendation(
            slug=model.index_to_film[index],
            url=f"https://letterboxd.com/film/{model.index_to_film[index]}",
            score=predictions[index],
        )
        for index in top_indices
    ]

    return recommendations


@app.get("/similar")
def similar(
    films: Annotated[list[str] | None, Query()], n: Optional[int] = 10
) -> List[FilmRecommendation]:
    """
    Recommend similar films

    :param List[str] films: A list of film slugs. If len(films) > 1, the mean of their
    embeddings will be used
    :param Optional[int] n: number of recommendations to make, defaults to 10
    :return List[FilmRecommendation]: films which are most similar to the target film(s)
    """
    if isinstance(films, str):
        films = [films]

    film_indices = []
    for film in films:
        if film in model.film_to_index:
            film_indices.append(model.film_to_index[film])
        else:
            raise ValueError(f"{film} isn't in our list of films!")

    embedding = (
        model.film_embeddings[film_indices].mean(dim=0).unsqueeze(0).detach().numpy()
    )

    predictions = cosine_similarity(model.film_embeddings, embedding)

    top_indices = np.argsort(predictions)[::-1][:n]
    recommendations = [
        FilmRecommendation(
            slug=model.index_to_film[index.item()],
            url=f"https://letterboxd.com/film/{model.index_to_film[index.item()]}",
            score=predictions[index],
        )
        for index in top_indices
    ]

    return recommendations
