import numpy as np
import torch
from src.recommender import Recommender
from src.dataset import LetterboxdDataset


def test_models_produce_same_user_embeddings():
    dummy_dataset = LetterboxdDataset.dummy(1000)
    torch_model = Recommender(dataset=dummy_dataset, embedding_size=10)
    torch_model.eval()
    numpy_model = torch_model.to_numpy()

    n = 5
    film_indices = torch.randint(0, len(dummy_dataset.films), (n,))
    ratings = torch.randint(1, 10, (n,)) / 2

    torch_results = torch_model.get_user_embedding_from_ratings(
        film_indices=film_indices, ratings=ratings.unsqueeze(1)
    )

    numpy_results = numpy_model.get_user_embedding_from_ratings(
        film_indices=film_indices.detach().numpy(), ratings=ratings.detach().numpy()
    )

    assert np.allclose(torch_results.detach().numpy(), numpy_results, atol=1e-5)


def test_models_produce_same_recommendations():
    dummy_dataset = LetterboxdDataset.dummy(1000)
    torch_model = Recommender(dataset=dummy_dataset, embedding_size=10)
    torch_model.eval()
    numpy_model = torch_model.to_numpy()

    n = 5
    user_indices = torch.randint(0, len(dummy_dataset.users), (n,))
    user_embeddings = torch_model.get_user_embeddings(user_indices=user_indices)
    film_embeddings = torch_model.film_embeddings.weight

    torch_results = torch_model.forward(
        user_embeddings=user_embeddings, film_embeddings=film_embeddings
    )
    numpy_results = numpy_model.predict(
        user_embeddings=user_embeddings.detach().numpy(),
        film_embeddings=film_embeddings.detach().numpy(),
    )

    assert np.allclose(torch_results.detach().numpy(), numpy_results, atol=1e-5)
