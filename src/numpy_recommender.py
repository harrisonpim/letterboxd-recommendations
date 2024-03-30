from typing import Optional
import numpy as np
import pickle


class NumpyRecommender:
    def __init__(
        self,
        film_slugs: np.ndarray,
        film_embeddings: np.ndarray,
        film_biases: np.ndarray,
        feed_forward_weights: np.ndarray,
        feed_forward_biases: np.ndarray,
    ):
        """
        Recommender system for films based on letterboxd user ratings. This is an
        inference-only, numpy implementation of the Recommender class. It is useful for
        deploying models to production environments where PyTorch is not available.

        :param np.ndarray film_slugs: The list of films in the dataset
        :param np.ndarray film_embeddings: The corresponding list of film embeddings
        :param float film_biases: The corresponding bias values for each film
        :param np.ndarray feed_forward_weights: The weights of the feed forward layers
        :param np.ndarray feed_forward_biases: The biases of the feed forward layers
        """
        assert (
            film_embeddings.shape[0] == len(film_slugs)
        ), f"number of film embeddings ({film_embeddings.shape[0]}) must match number of film slugs ({len(film_slugs)})"

        self.film_slugs = film_slugs
        self.film_embeddings = film_embeddings
        self.film_biases = film_biases

        self.feed_forward_weights = feed_forward_weights
        self.feed_forward_biases = feed_forward_biases

        self.film_to_index = {film: i for i, film in enumerate(film_slugs)}
        self.index_to_film = {i: film for i, film in enumerate(film_slugs)}

    def get_user_embedding_from_ratings(
        self, film_indices: np.ndarray, ratings: np.ndarray
    ) -> np.ndarray:
        """
        Generate an embedding for a user based on the films that they have rated

        :param np.ndarray film_indices: The indices of the films the user has rated
        :param np.ndarray ratings: The corresponding ratings for each film
        :return np.ndarray: The user embedding
        """
        weights = (ratings - ratings.mean()) + 0.1
        user_film_embeddings = (
            self.film_embeddings[film_indices] + self.film_biases[film_indices]
        )
        weighted_embeddings = user_film_embeddings * weights.reshape(-1, 1)
        concat_embeddings = np.concatenate(
            [
                weighted_embeddings.min(axis=0),
                weighted_embeddings.max(axis=0),
                weighted_embeddings.mean(axis=0),
                np.median(weighted_embeddings, axis=0),
            ]
        )
        concat_embeddings = concat_embeddings / np.linalg.norm(concat_embeddings)
        user_embedding = self.feed_forward_layers(concat_embeddings)
        return user_embedding

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """
        ReLU activation function

        :param np.ndarray x: The input to the activation function
        :return np.ndarray: The output of the activation function
        """
        return np.maximum(0, x)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute the cosine similarity between two groups of vectors.

        :param np.ndarray a: The first group of vectors. Shape: (a, m)
        :param np.ndarray b: The second group of vectors. Shape: (b, m)
        :return float: The cosine similarity between the two groups of vectors. Shape: (a, b)
        """
        a_norm = np.linalg.norm(a, axis=1)
        b_norm = np.linalg.norm(b, axis=1)
        dot_product = np.matmul(a, b.T)
        return dot_product / np.outer(a_norm, b_norm)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function

        :param np.ndarray x: The input to the activation function
        :return np.ndarray: The output of the activation function
        """
        return 1 / (1 + np.exp(-x))

    def feed_forward_layers(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the feed forward layers.

        :param np.ndarray x: The input to the feed forward layers
        :return np.ndarray: The output of the feed forward layers
        """
        for i, (weights, biases) in enumerate(
            zip(self.feed_forward_weights, self.feed_forward_biases)
        ):
            x = np.matmul(weights, x) + biases
            if i < len(self.feed_forward_weights) - 1:
                x = self.relu(x)
        return x

    def predict(
        self,
        user_embeddings: np.ndarray,  # size: (batch_size, embedding_size)
        film_embeddings: Optional[np.ndarray] = None,  # size: (n_films, embedding_size)
        diagonal_only: bool = True,
    ) -> np.ndarray:  # size: (batch_size, batch_size)
        if film_embeddings is None:
            film_embeddings = self.film_embeddings

        if diagonal_only:
            # only calculate the interactions for the diagonal of the matrix
            assert user_embeddings.shape == film_embeddings.shape
            A = user_embeddings.reshape(
                user_embeddings.shape[0], 1, user_embeddings.shape[1]
            )
            B = film_embeddings.reshape(
                1, film_embeddings.shape[0], film_embeddings.shape[1]
            )
            predictions = np.diagonal(np.sum(A * B, axis=-1), axis1=0, axis2=1)

        else:
            # calculate all interactions
            predictions = np.matmul(user_embeddings, film_embeddings.T).squeeze()

        return self.sigmoid(predictions)

    def save(self, path: str):
        """
        Save the model to a file.

        :param str path: The path to save the model to.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        """
        Load the model from a file.

        :param str path: The path to load the model from.
        :return Recommender: The loaded model.
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        assert isinstance(model, cls)
        return model
