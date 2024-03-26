from pathlib import Path

import pandas as pd
from humanize import intcomma
from rich.console import Console

console = Console(highlight=False)

data_dir = Path("data")
for file in sorted((data_dir / "raw").glob("*.json")):
    raw_data_path = data_dir / "raw" / file.name
    processed_data_path = data_dir / "processed" / file.name

    console.print(raw_data_path, style="bold white")
    df = pd.read_json(raw_data_path)
    n_rows_before = len(df)
    console.print(f"  Raw data has {intcomma(n_rows_before)} rows")

    # drop any duplicate rows
    df = df.drop_duplicates()

    # drop any film which hasn't been rated by more than 20 users
    df = df.groupby("film-slug").filter(lambda x: len(x) > 20)

    # drop any user who hasn't rated more than 3 films
    df = df.groupby("username").filter(lambda x: len(x) > 3)

    n_rows_after = len(df)
    console.print(f"  Dropped {intcomma(n_rows_before - n_rows_after)} rows")

    console.print(
        f"  Processed data has {intcomma(len(df))} rows, covering "
        f"{intcomma(len(df['username'].unique()))} users and "
        f"{intcomma(len(df['film-slug'].unique()))} films"
    )
    console.print(
        f"  Average rating is {df['rating'].mean():.2f}, with a standard deviation of "
        f"{df['rating'].std():.2f}"
    )

    # implement a train / test split
    # for each user, take 20% of their ratings and put them in the test set
    test_indexes = (
        df.groupby("username")
        .apply(lambda x: x.sample(frac=0.2, random_state=42).index)
        .reset_index(drop=True)
        .explode()
        .to_list()
    )

    df_train = df.drop(test_indexes)
    df_test = df.loc[test_indexes]

    # make sure that the test dataset only contains users and films that are in the
    # training dataset
    df_test = df_test[
        df_test["username"].isin(df_train["username"])
        & df_test["film-slug"].isin(df_train["film-slug"])
    ]

    console.print(
        f"  Training data has {intcomma(len(df_train))} rows, covering "
        f"{intcomma(len(df_train['username'].unique()))} users and "
        f"{intcomma(len(df_train['film-slug'].unique()))} films"
    )

    console.print(
        f"  Test data has {intcomma(len(df_test))} rows, covering "
        f"{intcomma(len(df_test['username'].unique()))} users and "
        f"{intcomma(len(df_test['film-slug'].unique()))} films"
    )

    # save the processed data
    (data_dir / "processed" / file.stem).mkdir(exist_ok=True)
    df.to_json(
        data_dir / "processed" / file.stem / "all.json", lines=True, orient="records"
    )
    df_train.to_json(
        data_dir / "processed" / file.stem / "train.json", lines=True, orient="records"
    )
    df_test.to_json(
        data_dir / "processed" / file.stem / "test.json", lines=True, orient="records"
    )

    df["film-slug"].sort_values().drop_duplicates().to_csv(
        data_dir / "processed" / file.stem / "films.csv", index=False, header=False
    )
    df["username"].sort_values().drop_duplicates().to_csv(
        data_dir / "processed" / file.stem / "users.csv", index=False, header=False
    )
