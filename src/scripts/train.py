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

data_dir = Path("data/processed/2023-11-10")
embedding_size = 10
batch_size = 512

train_dataset = LetterboxdDataset.read_json(data_dir / "train.json", n=100_000)
test_dataset = LetterboxdDataset.read_json(data_dir / "test.json", n=10_000)

console.print(
    f"The train dataset contains {len(train_dataset)} samples across {len(train_dataset.users)} users and {len(train_dataset.films)} films."
)
console.print(
    f"The test dataset contains {len(test_dataset)} samples across {len(test_dataset.users)} users and {len(test_dataset.films)} films."
)

model = Recommender(train_dataset, embedding_size=embedding_size)

train_dataloader = train_dataset.get_dataloader(batch_size=batch_size)
test_dataloader = test_dataset.get_dataloader(batch_size=batch_size)

optimizer = Adam(model.parameters(), lr=0.003)
loss_function = MSELoss()
n_epochs = 10


losses = []
for epoch in range(n_epochs):
    model.train()
    train_progress_bar = tqdm(train_dataloader)
    for batch in train_progress_bar:
        optimizer.zero_grad()
        predictions = model.forward(
            user_indices=batch["user_index"], film_indices=batch["film_index"]
        )
        loss = loss_function(predictions, batch["rating"].float())
        loss.backward()
        optimizer.step()
        losses.append(loss)
        rolling_loss = sum(losses[-20:]) / len(losses[-20:])
        train_progress_bar.set_description(
            f"Epoch {epoch + 1} | Loss: {rolling_loss:.4f}"
        )

    model.eval()
    test_progress_bar = tqdm(test_dataloader)
    for batch in test_progress_bar:
        predictions = model.forward(
            user_indices=batch["user_index"], film_indices=batch["film_index"]
        )
        loss = loss_function(predictions, batch["rating"].float())
        losses.append(loss)
        rolling_loss = sum(losses[-20:]) / len(losses[-20:])
        test_progress_bar.set_description(
            f"Epoch {epoch + 1} | Loss: {rolling_loss:.4f}"
        )
