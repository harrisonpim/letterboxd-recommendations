import torch
from torch.nn import Dropout, Embedding, Linear, Module, ReLU, Sequential, init

from src.dataset import LetterboxdDataset
from src.numpy_recommender import NumpyRecommender


class Recommender(Module):
    def __init__(
        self, dataset: LetterboxdDataset = LetterboxdDataset.empty(), embedding_dim=50
    ):
        """
        Recommender system for films based on letterboxd user ratings.

        :param LetterboxdDataset dataset: Dataset containing user ratings.
        :param int embedding_dim: Dimensionality of the film embeddings
        """
        super().__init__()
        self.dataset = dataset
        self.mean_rating = dataset.mean_rating

        self.film_embeddings = Embedding(
            len(self.dataset.films), embedding_dim, dtype=torch.float32
        )
        init.normal_(self.film_embeddings.weight, mean=0.0, std=0.01)

        self.film_biases = Embedding(len(self.dataset.films), 1, dtype=torch.float32)
        init.constant_(self.film_biases.weight, 0.0)

        self.feed_forward_layers = Sequential(
            Linear(4 * embedding_dim, 2 * embedding_dim),
            Dropout(0.2),
            ReLU(),
            Linear(2 * embedding_dim, 2 * embedding_dim),
            Dropout(0.2),
            ReLU(),
            Linear(2 * embedding_dim, 2 * embedding_dim),
            Dropout(0.2),
            ReLU(),
            Linear(2 * embedding_dim, 2 * embedding_dim),
            Dropout(0.2),
            ReLU(),
            Linear(2 * embedding_dim, embedding_dim),
        )
        for layer in self.feed_forward_layers:
            if isinstance(layer, Linear):
                init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
                init.constant_(layer.bias, 0.0)

    def get_user_embedding(self, user_index: int) -> torch.Tensor:
        """
        Get the embedding for a user.

        :param int user_index: The letterboxd user's index in the dataset
        :return torch.Tensor: The user embedding.
        """
        user_ratings = self.dataset.get_user_ratings(user_index)
        ratings = torch.tensor(
            user_ratings["rating"].values, dtype=torch.float32
        ).unsqueeze(1)
        film_indices = torch.tensor(
            user_ratings["film_index"].values, dtype=torch.int32
        )
        user_embedding = self.get_user_embedding_from_ratings(film_indices, ratings)
        return user_embedding

    def get_user_embedding_from_ratings(
        self, film_indices: torch.Tensor, ratings: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate an embedding for a user based on the films that they have rated

        User embeddings are formed by concatenating the min, max, mean, and median of
        the embeddings of the films that the user has rated. Building user embeddings
        from the film embeddings is a way to get a reasonable embedding for users who
        are not in the training set, ie overcoming the cold start problem.
        Concatenated embeddings are then passed through a small feed forward network to
        get the user embedding to match the size of the film embeddings, which are used
        to make predictions.

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
        weights = ratings - ratings.mean() + 0.1
        user_film_embeddings = self.film_embeddings(film_indices) + self.film_biases(
            film_indices
        )
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
        user_embeddings: torch.Tensor,  # size: (batch_size, embedding_dim)
        film_indices: torch.Tensor,  # size: (batch_size)
        diagonal_only: bool = True,
    ) -> torch.Tensor:  # size: (batch_size, batch_size)
        """
        Forward pass of the model, calculating the predicted ratings for each user-film
        interaction.

        :param torch.Tensor user_embeddings: A batch of user embeddings
        :param torch.Tensor film_indices: A batch of film indices
        :param bool diagonal_only: if True, only calculate the interactions for the
        diagonal of the matrix. This is useful for calculating the predictions for a set
        of known user-film interactions, where the other interactions can be ignored. If
        false, calculate the interactions for the full matrix. default: True
        :return torch.Tensor: Predicted ratings for each user-film interaction
        """
        film_embeddings = self.film_embeddings(film_indices) + self.film_biases(
            film_indices
        )
        if diagonal_only:
            # Only calculate the interactions for the diagonal of the matrix
            assert user_embeddings.shape == film_embeddings.shape
            A = user_embeddings.view(
                user_embeddings.shape[0], 1, user_embeddings.shape[1]
            )
            B = film_embeddings.view(
                1, film_embeddings.shape[0], film_embeddings.shape[1]
            )
            predictions = torch.diagonal(torch.sum(A * B, dim=-1), dim1=0, dim2=1)

        else:
            # Calculate all interactions
            predictions = torch.matmul(user_embeddings, film_embeddings.T).squeeze()

        return torch.sigmoid(predictions)

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

    def to_numpy(self) -> "NumpyRecommender":
        """
        Convert the model to a NumpyRecommender.

        :return NumpyRecommender: A NumpyRecommender version of the model.
        """
        film_embeddings = self.film_embeddings.weight.detach().numpy()
        film_biases = self.film_biases.weight.detach().numpy()
        feed_forward_weights, feed_forward_biases = zip(
            *[
                (layer.weight.detach().numpy(), layer.bias.detach().numpy())
                for layer in self.feed_forward_layers
                if isinstance(layer, Linear)
            ]
        )
        numpy_model = NumpyRecommender(
            film_slugs=self.dataset.films,
            film_embeddings=film_embeddings,
            film_biases=film_biases,
            feed_forward_weights=feed_forward_weights,
            feed_forward_biases=feed_forward_biases,
        )
        return numpy_model
