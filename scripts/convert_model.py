"""
Converts a model from its original pytorch format to a numpy format, so that it can be 
used for lightweight inference in a production environment.
"""

from pathlib import Path
import typer
from rich import console

from typing import Optional
from src.recommender import Recommender

console = console.Console(highlight=False)


def convert_model(model_path: str, output_path: Optional[str] = None):
    """
    _summary_

    :param str model_path: _description_
    :param Optional[str] output_path: _description_
    """
    model_path = Path(model_path)
    assert model_path.exists(), f"Model path {model_path} does not exist"
    if output_path is None:
        # use the same base path and file name as the model, but dump it in the /numpy folder
        output_dir = model_path.parent / "numpy"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / model_path.name

    console.print(f"Loading model from {model_path}")
    model = Recommender.load(model_path)
    console.print("Converting model to numpy format")
    numpy_model = model.to_numpy()
    console.print(f"Saving model to {output_path}")
    numpy_model.save(output_path)
    console.print("Model conversion complete")


if __name__ == "__main__":
    typer.run(convert_model)
