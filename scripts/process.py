import pandas as pd
from pathlib import Path

data_dir = Path("data")

# data should be saved in a format like data/raw/2023-11-10.json
# we want to pick the latest file
most_recent_filename = sorted((data_dir / "raw").glob("*.json"))[-1].name
raw_data_path = data_dir / "raw" / most_recent_filename
processed_data_path = data_dir / "processed" / most_recent_filename

df = pd.read_json(raw_data_path)

# drop any film which hasn't been rated by more than 3 users
df = df.groupby("film-slug").filter(lambda x: len(x) > 3)

# drop any user who hasn't rated more than 3 films
df = df.groupby("username").filter(lambda x: len(x) > 3)

# save the processed data as regular jsonl
df.to_json(processed_data_path, lines=True, orient="records")
