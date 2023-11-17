from typing import Union
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class LetterboxdDataset(Dataset):
    def __init__(self, ratings: pd.DataFrame):
        assert "username" in ratings.columns
        assert "film-slug" in ratings.columns
        assert "rating" in ratings.columns

        self.ratings = ratings
        self.users = self.ratings["username"].unique()
        self.movies = self.ratings["film-slug"].unique()

    @classmethod
    def read_json(cls, path: Union[Path, str]):
        ratings = pd.read_json(path, orient="records", lines=True)
        return cls(ratings)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.ratings[idx]

    def get_user_ratings(self, username):
        return self.ratings[self.ratings["username"] == username].to_dict("records")

    def get_movie_ratings(self, film_slug):
        return self.ratings[self.ratings["film-slug"] == film_slug].to_dict("records")

    def get_dataloader(self, batch_size, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
