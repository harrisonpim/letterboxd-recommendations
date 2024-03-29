import numpy as np
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class LetterboxdDataset(Dataset):
    def __init__(
        self,
        ratings: pd.DataFrame,
        n: Optional[int] = None,
        use_indices_from: "LetterboxdDataset" = None,
    ):
        assert "username" in ratings.columns
        assert "film-slug" in ratings.columns
        assert "rating" in ratings.columns

        self.ratings = ratings
        if n is not None:
            self.ratings = self.ratings.sample(n=n)
        self.users = self.ratings["username"].unique()
        self.films = self.ratings["film-slug"].unique()
        self.mean_rating = self.ratings["rating"].mean()

        if use_indices_from is not None:
            self.user_to_index = use_indices_from.user_to_index
            self.index_to_user = use_indices_from.index_to_user
            self.film_to_index = use_indices_from.film_to_index
            self.index_to_film = use_indices_from.index_to_film
        else:
            self.user_to_index = {user: i for i, user in enumerate(self.users)}
            self.index_to_user = {i: user for user, i in self.user_to_index.items()}
            self.film_to_index = {film: i for i, film in enumerate(self.films)}
            self.index_to_film = {i: film for film, i in self.film_to_index.items()}

        self.ratings["user_index"] = self.ratings["username"].map(self.user_to_index)
        self.ratings["film_index"] = self.ratings["film-slug"].map(self.film_to_index)
        self.ratings["rating"] = self.ratings["rating"] / 5
        self.ratings = self.ratings[["user_index", "film_index", "rating"]]

    @classmethod
    def read_json(
        cls,
        path: Union[Path, str],
        n: Optional[int] = None,
        use_indices_from: "LetterboxdDataset" = None,
    ) -> "LetterboxdDataset":
        """
        Read a JSON file containing Letterboxd ratings, with one rating per line.

        The JSON file should have the following format:

        ```json
        {
            "username": "username",
            "film-slug": "film-slug",
            "rating": 5
        }
        ```

        :param Union[Path, str] path: The path to the JSON file.
        :param Optional[int] n: An optional number of ratings to sample.
        :param LetterboxdDataset use_indices_from: An optional dataset to use the
        indices from. This is useful when you want to use the same indices for the
        train and test datasets.
        :return LetterboxdDataset: A dataset containing the ratings.
        """
        ratings = pd.read_json(path, orient="records", lines=True)
        return cls(ratings, n=n, use_indices_from=use_indices_from)

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int) -> tuple[int, int, float]:
        row = self.ratings.iloc[idx]
        return (
            torch.tensor(row["user_index"]).int(),
            torch.tensor(row["film_index"]).int(),
            torch.tensor(row["rating"]).float(),
        )

    def get_user_ratings(self, user_index: int) -> pd.DataFrame:
        """
        Get the ratings for a single user.

        :param int user_index: The index of the user in the dataset.
        :return pd.DataFrame: The user's ratings.
        """
        return self.ratings[self.ratings["user_index"] == int(user_index)]

    def get_film_ratings(self, film_index: int) -> pd.DataFrame:
        """
        Get the ratings for a single film.

        :param int film_index: The index of the film in the dataset.
        :return pd.DataFrame: The film's ratings.
        """
        return self.ratings[self.ratings["film_index"] == film_index]

    def get_dataloader(self, batch_size: int, shuffle: bool = True) -> DataLoader:
        """
        Returns a pytorch DataLoader for the dataset, with the given batch size.

        :param int batch_size: The number of samples per batch.
        :param bool shuffle: If true, the data is shuffled before each epoch. Defaults
        to True.
        :return DataLoader: A pytorch DataLoader with the given batch size.
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    @classmethod
    def empty(cls) -> "LetterboxdDataset":
        """
        Create an empty dataset.
        """
        return cls(pd.DataFrame(columns=["username", "film-slug", "rating"]))

    @classmethod
    def dummy(cls, n: int) -> "LetterboxdDataset":
        """
        Create a dummy dataset with n ratings. The number of users should be roughly
        sqrt(n). The films are named "film-{i}" where i is a random integer between 0
        and sqrt(n). The ratings are random floats between 0 and 5 in increments of 0.5.
        """
        dummy_df = pd.DataFrame(
            {
                "username": [f"user-{int(i**0.5)}" for i in range(n)],
                "film-slug": [
                    f"film-{np.random.randint(0, int(n**0.5))}" for _ in range(n)
                ],
                "rating": np.random.randint(0, 10, n) / 2,
            }
        )
        return cls(dummy_df)
