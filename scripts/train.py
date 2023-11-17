from pathlib import Path
from src.model import Recommender
from src.dataset import LetterboxdDataset
from torch.optim import Adam
from torch.nn import MSELoss

data_dir = Path("data/processed/2023-11-10")

train_dataset = LetterboxdDataset.read_json(data_dir / "train.json")
test_dataset = LetterboxdDataset.read_json(data_dir / "test.json")

model = Recommender(train_dataset, embedding_size=50)

train_dataloader = train_dataset.get_dataloader(batch_size=32)
test_dataloader = test_dataset.get_dataloader(batch_size=32)

optimizer = Adam(model.parameters(), lr=0.001)
loss_function = MSELoss()
n_epochs = 10
for epoch in range(n_epochs):
    for batch in train_dataloader:
        optimizer.zero_grad()
        predictions = model.forward(users=batch["username"], movies=batch["film-slug"])
        loss = loss_function(predictions, batch["rating"])
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: {loss.item()}")
