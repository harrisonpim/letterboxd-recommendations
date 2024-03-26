from typing import Optional

import pandas as pd
import torch
from torch.nn import Dropout, Embedding, Linear, Module, ReLU, Sequential

from src.dataset import LetterboxdDataset


class Recommender(Module):
    def __init__(
        self, dataset: LetterboxdDataset = LetterboxdDataset.empty(), embedding_size=50
    ):
        """
        Recommender system for films based on letterboxd user ratings.

        :param LetterboxdDataset dataset: Dataset containing user ratings.
        :param int embedding_size: Size of the film embeddings. User embeddings are 4
        times this size.
        """
        super().__init__()
        self.dataset = dataset
        self.mean_rating = dataset.mean_rating

        self.film_embeddings = Embedding(
            len(self.dataset.films), embedding_size, dtype=torch.float32
        )

        self.feed_forward_layers = Sequential(
            Linear(4 * embedding_size, 2 * embedding_size),
            Dropout(0.2),
            ReLU(),
            Linear(2 * embedding_size, 2 * embedding_size),
            Dropout(0.2),
            ReLU(),
            Linear(2 * embedding_size, embedding_size),
        )

    def get_user_embedding(self, user_index: int) -> torch.Tensor:
        """
        Get the embedding for a user.

        :param int user_index: The letterboxd user's index in the dataset
        :return torch.Tensor: The user embedding.
        """

        user_ratings = self.dataset.get_user_ratings(user_index)
        ratings = torch.tensor(user_ratings["rating"].values).unsqueeze(1)
        film_indices = torch.tensor(user_ratings["film_index"].values)
        user_embedding = self.get_user_embedding_from_ratings(film_indices, ratings)
        return user_embedding

    def get_user_embedding_from_ratings(
        self, film_indices: torch.Tensor, ratings: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate an embedding for a user based on the films that they have rated

        User embeddings are the concatenated min, max, mean, and median of the
        embeddings of the films that the user has rated. Building user embeddings from
        the film embeddings is a way to get a reasonable embedding for users who are
        not in the training set, ie overcoming the cold start problem.

        Concatenated embeddings are then passed through a feed forward network to get
        the user embedding to match the size of the film embeddings.

        To account for users' preferences within the set of films they have rated,
        the constituent film embeddings are weighted by their rating. Weighting the
        embeddings by the rating is equivalent to weighting the embeddings by the
        difference between the rating and the mean rating for that user. Some of the
        ratings are negative, so the ratings are shifted so that films which are given
        the mean rating for that user have a weight of 0.1

        :param torch.Tensor film_indices: The indices of the films the user has rated
        :param torch.Tensor ratings: The corresponding ratings for each film
        :return torch.Tensor: The user embedding
        """
        # TODO account for each user's mean rating, rather than global mean
        weights = (ratings - self.mean_rating) + 0.1
        user_film_embeddings = self.film_embeddings(film_indices)
        weighted_embeddings = user_film_embeddings * weights
        concat_embeddings = torch.cat(
            [
                weighted_embeddings.min(dim=0)[0].to(torch.float32),
                weighted_embeddings.max(dim=0)[0].to(torch.float32),
                weighted_embeddings.mean(dim=0).to(torch.float32),
                weighted_embeddings.median(dim=0)[0].to(torch.float32),
            ]
        )
        concat_embeddings = concat_embeddings / concat_embeddings.norm()
        user_embedding = self.feed_forward_layers(concat_embeddings)
        return user_embedding

    def get_user_embeddings(self, user_indices: torch.Tensor) -> torch.Tensor:
        """
        Get the embeddings for a batch of users.

        :param torch.Tensor user_indices: A batch of user indices.
        :return torch.Tensor: The user embeddings.
        """
        return torch.stack(
            [self.get_user_embedding(user_id) for user_id in user_indices]
        )

    def forward(
        self,
        user_indices: torch.Tensor,  # size: (batch_size, 1)
        film_indices: torch.Tensor,  # size: (batch_size, 1)
    ) -> torch.Tensor:  # size: (batch_size, batch_size)
        """
        Forward pass of the model.

        :param torch.Tensor user_indices: A batch of user indexes
        :param torch.Tensor film_indices: A batch of film indexes
        :return torch.Tensor: Predicted ratings
        """
        user_embeddings = self.get_user_embeddings(user_indices)
        film_embeddings = self.film_embeddings(film_indices.int())
        return torch.matmul(user_embeddings, film_embeddings.T)

    def predict(
        self, user_indices: torch.Tensor, film_indices: Optional[torch.Tensor] = None
    ) -> pd.DataFrame:
        """
        Predict the rating that a user will give to a film.

        :param torch.Tensor user_indices: The user indices.
        :param torch.Tensor film_indices: The film indices. If None, the model
        will predict the rating for all films in the dataset.
        :return pd.DataFrame: A dataframe containing the predicted ratings for each
        film.
        """
        if film_indices is None:
            film_indices = torch.arange(len(self.dataset.films))

        film_slugs = [self.dataset.index_to_film[i] for i in film_indices.numpy()]

        full_predicted_scores = (
            self.forward(user_indices, film_indices).detach().numpy()
        )

        # just take the diagonal values of the matrix. These are the interactions between
        # user i and film i, ie the pairings that we have ratings for
        predictions = torch.diag(full_predicted_scores)

        return pd.DataFrame({"film-slug": film_slugs, "predicted-rating": predictions})

    def save(self, path: str):
        """
        Save the model to a file.

        :param str path: The path to save the model to.
        """
        torch.save(self, path)

    @classmethod
    def load(cls, path: str):
        """
        Load the model from a file.

        :param str path: The path to load the model from.
        :return Recommender: The loaded model.
        """
        model = torch.load(path)
        assert isinstance(model, cls)
        model.eval()
        return model
