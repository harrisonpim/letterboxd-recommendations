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
        Recommender system for movies based on letterboxd user ratings.

        :param LetterboxdDataset dataset: Dataset containing user ratings.
        :param int embedding_size: Size of the movie embeddings. User embeddings are 4
        times this size.
        """
        super().__init__()
        self.dataset = dataset

        self.user_to_index = {user: i for i, user in enumerate(self.dataset.users)}
        self.index_to_user = {i: user for user, i in self.user_to_index.items()}

        self.movie_to_index = {movie: i for i, movie in enumerate(self.dataset.movies)}
        self.index_to_movie = {i: movie for movie, i in self.movie_to_index.items()}

        self.movie_embeddings = Embedding(len(self.dataset.movies), embedding_size)

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

    def get_user_embedding(self, user_id: str) -> torch.Tensor:
        """
        Get the embedding for a user.

        User embeddings are the concatenated min, max, mean, and median of the
        embeddings of the movies that the user has rated. Building user embeddings from
        the movie embeddings is a way to get a reasonable embedding for users who are
        not in the training set, ie overcoming the cold start problem.

        To account for users' preferences within the set of films they have rated,
        the constituent movie embeddings are weighted by their rating.

        :param str user_id: The letterboxd username, used to look up the user's ratings
        in the dataset.
        :return torch.Tensor: The user embedding.
        """
        user_ratings = self.dataset.get_user_ratings(user_id)["film-slug"]
        scores = torch.tensor(user_ratings["rating"], dtype=torch.float32).unsqueeze(1)
        movie_indices = [
            self.movie_to_index[slug] for slug in user_ratings["film-slug"]
        ]
        user_movie_embeddings = self.movie_embeddings(movie_indices)
        weighted_embeddings = user_movie_embeddings * scores
        user_embedding = torch.cat(
            [
                weighted_embeddings.min(dim=0),
                weighted_embeddings.max(dim=0),
                weighted_embeddings.mean(dim=0),
                weighted_embeddings.median(dim=0),
            ]
        )
        return user_embedding / user_embedding.norm()

    def get_user_embeddings(self, user_ids: list[str]) -> torch.Tensor:
        """
        Get the embeddings for a batch of users.

        :param list[str] user_ids: A batch of user ids.
        :return torch.Tensor: The user embeddings.
        """
        return torch.stack([self.get_user_embedding(user_id) for user_id in user_ids])

    def forward(self, users: list[str], movies: list[str]) -> torch.Tensor:
        """
        Forward pass of the model.

        :param list[str] users: User ids.
        :param list[str] movies: Movie ids.
        :return torch.Tensor: Predicted ratings.
        """
        user_embeddings = self.get_user_embeddings(users)
        movie_embeddings = self.movie_embeddings(movies)
        input_embeddings = torch.cat([user_embeddings, movie_embeddings], dim=1)
        return self.feed_forward_layers(input_embeddings)

    def predict(self, users: list[str], movies: list[str]) -> torch.Tensor:
        """
        Predict ratings for a batch of users and movies.

        :param list[str] users: User ids.
        :param list[str] movies: Movie ids.
        :return torch.Tensor: Predicted ratings.
        """
        with torch.no_grad():
            return self.forward(users, movies)
