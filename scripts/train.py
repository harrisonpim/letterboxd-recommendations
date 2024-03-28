import warnings
from datetime import datetime
from pathlib import Path

import torch
from rich.console import Console
from torch.nn import MSELoss
from torch.optim import Adam
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

from src.dataset import LetterboxdDataset
from src.model import Recommender

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
console = Console(highlight=False)

data_dir = Path("data/processed/2023-11-19")
model_dir = Path("data/models/")
model_dir.mkdir(exist_ok=True)

iso_date = datetime.now().strftime("%Y-%m-%d")
model_path = model_dir / f"{iso_date}.pkl"

train_dataset = LetterboxdDataset.read_json(data_dir / "train.json")
test_dataset = LetterboxdDataset.read_json(
    data_dir / "test.json", use_indices_from=train_dataset
)

console.print(
    f"The train dataset contains {len(train_dataset)} samples across "
    f"{len(train_dataset.users)} users and {len(train_dataset.films)} films."
)
console.print(
    f"The test dataset contains {len(test_dataset)} samples across "
    f"{len(test_dataset.users)} users and {len(test_dataset.films)} films."
)

embedding_size = 10
model = Recommender(train_dataset, embedding_size=embedding_size)

batch_size = 512
train_dataloader = train_dataset.get_dataloader(batch_size=batch_size)
test_dataloader = test_dataset.get_dataloader(batch_size=batch_size)

optimizer = Adam(model.parameters(), lr=0.003)
loss_function = MSELoss()

n_epochs = 3
for epoch in range(n_epochs):
    train_losses = []
    model.train()
    train_progress_bar = tqdm(train_dataloader)
    for i, batch in enumerate(train_progress_bar):
        optimizer.zero_grad()
        user_indices, film_indices, true_ratings = batch
        user_embeddings = model.get_user_embeddings(user_indices)
        film_embeddings = model.film_embeddings(film_indices)
        full_predicted_scores = model(user_embeddings, film_embeddings).squeeze()
        # just take the diagonal values of the matrix. These are the interactions between
        # user i and film i, ie the pairings we have ratings for
        predictions = torch.diag(full_predicted_scores)
        loss = loss_function(predictions, true_ratings.float())
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        rolling_loss = sum(train_losses[-20:]) / len(train_losses[-20:])
        train_progress_bar.set_description(
            f"Training epoch {epoch + 1}/{n_epochs} | Loss: {rolling_loss:.4f}"
        )
        if i % 100 == 0:
            model.save(model_path)

    test_losses = []
    model.eval()
    test_progress_bar = tqdm(test_dataloader)
    for batch in test_progress_bar:
        user_indices, film_indices, true_ratings = batch
        user_embeddings = model.get_user_embeddings(user_indices)
        film_embeddings = model.film_embeddings(film_indices)
        full_predicted_scores = model(user_embeddings, film_embeddings).squeeze()
        # just take the diagonal values of the matrix. These are the interactions between
        # user i and film i, ie the pairings we have ratings for
        predictions = torch.diag(full_predicted_scores)
        loss = loss_function(predictions, true_ratings.float())
        test_losses.append(loss)
        rolling_loss = sum(test_losses[-20:]) / len(test_losses[-20:])
        test_progress_bar.set_description(
            f"Testing epoch {epoch + 1}/{n_epochs} | Loss: {rolling_loss:.4f}"
        )

    model.save(model_path)
    console.print(f"Saved model to {model_path}")
