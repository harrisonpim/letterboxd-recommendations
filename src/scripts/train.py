from datetime import datetime
from pathlib import Path
from src.model import Recommender
from src.dataset import LetterboxdDataset
from torch.optim import Adam
from torch.nn import MSELoss
from rich.console import Console
from tqdm.rich import tqdm
import warnings
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
console = Console(highlight=False)

iso_date = datetime.now().strftime("%Y-%m-%d")
model_dir = Path("data/models/")
model_path = model_dir / f"{iso_date}.pkl"

data_dir = Path("data/processed/2023-11-10")
embedding_size = 10
batch_size = 512

train_dataset = LetterboxdDataset.read_json(data_dir / "train.json", n=20_000)
test_dataset = LetterboxdDataset.read_json(
    data_dir / "test.json", use_indices_from=train_dataset, n=5_000
)

console.print(
    f"The train dataset contains {len(train_dataset)} samples across "
    f"{len(train_dataset.users)} users and {len(train_dataset.films)} films."
)
console.print(
    f"The test dataset contains {len(test_dataset)} samples across "
    f"{len(test_dataset.users)} users and {len(test_dataset.films)} films."
)

model = Recommender(train_dataset, embedding_size=embedding_size)

train_dataloader = train_dataset.get_dataloader(batch_size=batch_size)
test_dataloader = test_dataset.get_dataloader(batch_size=batch_size)

optimizer = Adam(model.parameters(), lr=0.003)
loss_function = MSELoss()
n_epochs = 3


for epoch in range(n_epochs):
    train_losses = []
    model.train()

    train_progress_bar = tqdm(train_dataloader)
    for batch in train_progress_bar:
        user_indices, film_indices, ratings = batch
        optimizer.zero_grad()
        predictions = model.forward(
            user_indices=user_indices,
            film_indices=film_indices,
        )
        loss = loss_function(predictions.squeeze(), ratings.float())
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        rolling_loss = sum(train_losses[-20:]) / len(train_losses[-20:])
        train_progress_bar.set_description(
            f"Training epoch {epoch + 1}/{n_epochs} | Loss: {rolling_loss:.4f}"
        )

    test_losses = []
    model.eval()
    test_progress_bar = tqdm(test_dataloader)
    for batch in test_progress_bar:
        user_indices, film_indices, ratings = batch
        predictions = model.forward(
            user_indices=user_indices,
            film_indices=film_indices,
        )
        loss = loss_function(predictions.squeeze(), ratings.float())
        test_losses.append(loss)
        rolling_loss = sum(test_losses[-20:]) / len(test_losses[-20:])
        test_progress_bar.set_description(
            f"Testing epoch {epoch + 1}/{n_epochs} | Loss: {rolling_loss:.4f}"
        )

    model.save(model_path)
    console.print(f"Saved model to {model_path}")
