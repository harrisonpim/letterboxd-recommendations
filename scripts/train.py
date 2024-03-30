import warnings
from datetime import datetime
from pathlib import Path

from rich.console import Console
from torch.nn import MSELoss
from torch.optim import Adam
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm
from omegaconf import OmegaConf
from src.dataset import LetterboxdDataset
from src.recommender import Recommender

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
console = Console(highlight=False)

config = OmegaConf.load("training_config.yaml")
existing_model_path = config.model.get("existing_model_path")
embedding_dim = config.model.get("embedding_dim")
batch_size = config.training.get("batch_size")
n_epochs = config.training.get("n_epochs")
learning_rate = config.training.get("learning_rate", 0.001)
weight_decay = config.training.get("weight_decay", 0.0001)

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

if not existing_model_path:
    if not embedding_dim:
        raise ValueError(
            "Embedding size must be specified in config file if no model is provided."
        )
    model = Recommender(train_dataset, embedding_dim=embedding_dim)
else:
    model_path = Path(existing_model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file {model_path} not found.")
    model = Recommender.load(existing_model_path)
    console.print(f"Loaded model from {existing_model_path}")

if not batch_size:
    raise ValueError("Batch size must be specified in config file")

train_dataloader = train_dataset.get_dataloader(batch_size=batch_size)
test_dataloader = test_dataset.get_dataloader(batch_size=batch_size)


optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_function = MSELoss()

if not n_epochs:
    raise ValueError("Number of epochs must be specified in config file")

for epoch in range(n_epochs):
    train_losses = []
    model.train()
    train_progress_bar = tqdm(train_dataloader)
    for i, batch in enumerate(train_progress_bar):
        optimizer.zero_grad()
        user_indices, film_indices, true_ratings = batch
        user_embeddings = model.get_user_embeddings(user_indices)
        predictions = model(
            user_embeddings=user_embeddings,
            film_indices=film_indices,
            diagonal_only=True,
        )
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
        predictions = model(
            user_embeddings=user_embeddings,
            film_indices=film_indices,
            diagonal_only=True,
        )
        loss = loss_function(predictions, true_ratings.float())
        test_losses.append(loss)
        rolling_loss = sum(test_losses[-20:]) / len(test_losses[-20:])
        test_progress_bar.set_description(
            f"Testing epoch {epoch + 1}/{n_epochs} | Loss: {rolling_loss:.4f}"
        )

    model.save(model_path)
    console.print(f"Saved model to {model_path}")
