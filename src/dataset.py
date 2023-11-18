from typing import Union, Optional
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class LetterboxdDataset(Dataset):
    def __init__(self, ratings: pd.DataFrame, n: Optional[int] = None):
        assert "username" in ratings.columns
        assert "film-slug" in ratings.columns
        assert "rating" in ratings.columns

        self.ratings = ratings
        if n is not None:
            self.ratings = self.ratings.sample(n=n)
        self.users = self.ratings["username"].unique()
        self.films = self.ratings["film-slug"].unique()

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
        cls, path: Union[Path, str], n: Optional[int] = None
    ) -> "LetterboxdDataset":
        ratings = pd.read_json(path, orient="records", lines=True)
        return cls(ratings, n=n)

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int) -> dict:
        row = self.ratings.iloc[idx]
        return row.to_dict()

    def get_user_ratings(self, user_index: int) -> pd.DataFrame:
        return self.ratings[self.ratings["user_index"] == user_index]

    def get_film_ratings(self, film_index: int) -> pd.DataFrame:
        return self.ratings[self.ratings["film_index"] == film_index]

    def get_dataloader(self, batch_size: int, shuffle: bool = True) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
