import numpy as np
from typing import List, Optional, Annotated
from pydantic import BaseModel
from fastapi import FastAPI, Query
from src.numpy_recommender import NumpyRecommender
from src.letterboxd import stars_to_rating
import httpx
from pathlib import Path
import scrapy

model_dir = Path("data/models/numpy/")
model_path = sorted(model_dir.glob("*.pkl"))[-1]
model = NumpyRecommender.load(model_path)

app = FastAPI()


class Recommendation(BaseModel):
    slug: str
    url: str
    score: Optional[float] = None


@app.get("/predict")
def predict(
    username: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100, alias="pageSize"),
) -> List[Recommendation]:
    """
    Make predictions for the user based on their most recently watched films

    :param str username: The target username. Their recently watched films will be
    scraped and used to form a basis for making recommendations
    :param int page: The page number, defaults to 1
    :param int page_size: The number of films per page, defaults to 10
    :return List[Recommendation]: A list of films which the user is likely to enjoy
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
    # unsqueeze to add a batch dimension
    user_embeddings = np.expand_dims(user_embedding, axis=0)
    film_embeddings = model.film_embeddings

    # get the recommendations
    predictions = model.predict(user_embeddings, film_embeddings, diagonal_only=False)

    start_index = (page - 1) * page_size
    end_index = start_index + page_size
    indices = np.argsort(predictions)[::-1][start_index:end_index]

    recommendations = [
        Recommendation(
            slug=model.index_to_film[index],
            url=f"https://letterboxd.com/film/{model.index_to_film[index]}",
            score=predictions[index],
        )
        for index in indices
    ]

    return recommendations


@app.get("/similar")
def similar(
    films: Annotated[list[str] | None, Query()],
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100, alias="pageSize"),
) -> List[Recommendation]:
    """
    Recommend similar films

    :param List[str] films: A list of film slugs. If len(films) > 1, the mean of their
    embeddings will be used
    :param int page: The page number, defaults to 1
    :param int page_size: The number of films per page, defaults to 10
    :return List[Recommendation]: films which are most similar to the target film(s)
    """
    if isinstance(films, str):
        films = [films]

    film_indices = []
    for film in films:
        if film in model.film_to_index:
            film_indices.append(model.film_to_index[film])
        else:
            raise ValueError(f"{film} isn't in our list of films!")

    embedding = model.film_embeddings[film_indices].mean(axis=0).reshape(1, -1)
    predictions = model.cosine(model.film_embeddings, embedding).squeeze()

    start_index = (page - 1) * page_size
    end_index = start_index + page_size
    indices = np.argsort(predictions)[::-1][start_index:end_index]

    recommendations = [
        Recommendation(
            slug=model.index_to_film[index],
            url=f"https://letterboxd.com/film/{model.index_to_film[index]}",
            score=predictions[index],
        )
        for index in indices
    ]

    return recommendations


@app.get("/films")
def films(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100, alias="pageSize"),
) -> List[Recommendation]:
    """
    Get a paginated list of films in the dataset

    :param int page: The page number, defaults to 1
    :param int page_size: The number of films per page, defaults to 10
    :return List[str]: A list of film slugs for the requested page
    """
    start_index = (page - 1) * page_size
    end_index = start_index + page_size
    film_slugs = list(model.film_to_index.keys())[start_index:end_index]
    return [
        {
            "slug": slug,
            "url": f"https://letterboxd.com/film/{slug}",
        }
        for slug in film_slugs
    ]
