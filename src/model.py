from torch.nn import Module, Embedding, Sequential, Linear, ReLU, Sigmoid, Dropout
import torch
from src.dataset import LetterboxdDataset


class Recommender(Module):
    def __init__(
        self,
        dataset: LetterboxdDataset,
        embedding_size=50,
    ):
        """
        Recommender system for films based on letterboxd user ratings.

        :param LetterboxdDataset dataset: Dataset containing user ratings.
        :param int embedding_size: Size of the film embeddings. User embeddings are 4
        times this size.
        """
        super().__init__()
        self.dataset = dataset

        self.film_embeddings = Embedding(
            len(self.dataset.films), embedding_size, dtype=torch.float32
        )

        self.feed_forward_layers = Sequential(
            Linear(5 * embedding_size, 2 * embedding_size),
            Dropout(0.2),
            ReLU(),
            Linear(2 * embedding_size, embedding_size),
            Dropout(0.2),
            ReLU(),
            Linear(embedding_size, 1),
            Sigmoid(),
        )

    def get_user_embedding(self, user_index: int) -> torch.Tensor:
        """
        Get the embedding for a user.

        User embeddings are the concatenated min, max, mean, and median of the
        embeddings of the films that the user has rated. Building user embeddings from
        the film embeddings is a way to get a reasonable embedding for users who are
        not in the training set, ie overcoming the cold start problem.

        To account for users' preferences within the set of films they have rated,
        the constituent film embeddings are weighted by their rating.

        :param int user_index: The letterboxd user's index in the dataset
        :return torch.Tensor: The user embedding.
        """
        user_index = int(user_index)
        user_ratings = self.dataset.get_user_ratings(user_index)
        scores = torch.tensor(user_ratings["rating"].values).unsqueeze(1)
        film_indices = torch.tensor(user_ratings["film_index"].values)
        user_film_embeddings = self.film_embeddings(film_indices)
        weighted_embeddings = user_film_embeddings * scores
        user_embedding = torch.cat(
            [
                weighted_embeddings.min(dim=0)[0],
                weighted_embeddings.max(dim=0)[0],
                weighted_embeddings.mean(dim=0),
                weighted_embeddings.median(dim=0)[0],
            ]
        )
        return user_embedding / user_embedding.norm()

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
        self, user_indices: torch.Tensor, film_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        :param torch.Tensor user_indices: A batch of user indexes
        :param torch.Tensor film_indices: A batch of film indexes
        :return torch.Tensor: Predicted ratings
        """
        user_embeddings = self.get_user_embeddings(user_indices).to(torch.float32)
        film_embeddings = self.film_embeddings(film_indices.int()).to(torch.float32)
        input_embeddings = torch.cat([user_embeddings, film_embeddings], dim=1)
        return self.feed_forward_layers(input_embeddings).squeeze()

    def predict(self, user_indices: list[str], film_indices: list[str]) -> torch.Tensor:
        """
        Predict ratings for a batch of user_indices and film_indices.

        :param list[str] user_indices: A batch of user indexes
        :param list[str] film_indices: A batch of film indexes
        :return torch.Tensor: Predicted ratings
        """
        with torch.no_grad():
            return self.forward(user_indices, film_indices)
