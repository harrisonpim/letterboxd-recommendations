from rich.console import Console
import pandas as pd
from pathlib import Path
from humanize import intcomma

console = Console(highlight=False)

data_dir = Path("data")
for file in sorted((data_dir / "raw").glob("*.json")):
    raw_data_path = data_dir / "raw" / file.name
    processed_data_path = data_dir / "processed" / file.name

    console.print(raw_data_path, style="bold white")
    df = pd.read_json(raw_data_path)
    n_rows_before = len(df)
    console.print(f"  Raw data has {intcomma(n_rows_before)} rows")

    # drop any film which hasn't been rated by more than 3 users
    df = df.groupby("film-slug").filter(lambda x: len(x) > 3)

    # drop any user who hasn't rated more than 3 films
    df = df.groupby("username").filter(lambda x: len(x) > 3)

    n_rows_after = len(df)
    console.print(f"  Dropped {intcomma(n_rows_before - n_rows_after)} rows")

    # save the processed data as regular jsonl
    df.to_json(processed_data_path, lines=True, orient="records")

    console.print(
        f"  Processed data has {intcomma(len(df))} rows, covering "
        f"{intcomma(len(df['username'].unique()))} users and "
        f"{intcomma(len(df['film-slug'].unique()))} films"
    )
    console.print(
        f"  Average rating is {df['rating'].mean():.2f}, with a standard deviation of "
        f"{df['rating'].std():.2f}"
    )

    console.print(f"  Saved processed data to [bold]{processed_data_path}", "\n")
