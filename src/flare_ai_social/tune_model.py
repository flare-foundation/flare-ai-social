"""
Module for training and tuning AI models using Google's Generative AI API.
Handles model creation, training, and performance visualization.
"""

import json
from pathlib import Path
from typing import TypedDict

import google.generativeai as genai
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import structlog

from flare_ai_social.settings import settings

logger = structlog.get_logger(__name__)
genai.configure(api_key=settings.gemini_api_key)
sns.set_style("darkgrid")


class TrainingEntry(TypedDict):
    text_input: str
    output: str


TrainingData = list[TrainingEntry]


def get_tuning_supported_models() -> list[str]:
    """
    Retrieve list of models that support fine-tuning.

    Returns:
        list[str]: Names of models supporting tuning
    """
    models = [
        m.name
        for m in genai.list_models()
        if "createTunedModel" in m.supported_generation_methods
    ]
    logger.info("models supporting tuning identified", models=models)
    return models


def delete_existing_model(model_id: str) -> None:
    """
    Delete a tuned model if it exists.

    Args:
        model_id (str): ID of the model to delete
    """
    full_model_id = f"tunedModels/{model_id}"
    for tuned_model in genai.list_tuned_models():
        if tuned_model.name == full_model_id:
            logger.info("deleting existing model", tuned_model_id=model_id)
            genai.delete_tuned_model(full_model_id)
            break


def load_training_data(path: Path) -> TrainingData:
    """
    Load training dataset from a JSON file.

    Args:
        path (Path): Path to the training dataset file

    Returns:
        TrainingData: Loaded training dataset

    Raises:
        JSONDecodeError: If JSON file is invalid
        FileNotFoundError: If file doesn't exist
    """
    try:
        with path.open() as f:
            data = json.load(f)
        min_dataset_size = 20
        if len(data) < min_dataset_size:
            logger.warning(
                "small dataset, tuning quality may be poor",
                dataset_size=len(data),
                min_dataset_size=min_dataset_size,
            )
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.Exception("failed to load training data", error=str(e))
        raise
    else:
        return data


def save_loss_plot(
    snapshots: pd.DataFrame, model_id: str, save_path: str | None = None
) -> Path:
    """
    Create and save a plot of mean loss over epochs.

    Args:
        snapshots (pd.DataFrame): Training snapshots data
        model_id (str): ID of the model
        save_path (Optional[str]): Custom save path for the plot

    Returns:
        Path: Path where the plot was saved
    """
    fig, ax = plt.subplots()
    sns.set_style("darkgrid")

    sns.lineplot(data=snapshots, x="epoch", y="mean_loss", ax=ax)
    ax.set_title(f"Training Loss - {model_id}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Loss")

    save_path = save_path or f"{model_id}_mean_loss.png"
    fig.savefig(save_path)
    plt.close(fig)

    return Path(save_path)


def start() -> None:
    """
    Train a new model with the specified ID.

    Args:
        new_model_id (str): ID for the new model

    Raises:
        Exception: If model training fails
    """
    new_model_id = settings.tuned_model_name
    # Delete existing model if present
    delete_existing_model(new_model_id)

    # Load and validate training data
    training_dataset = load_training_data(settings.tuning_dataset_path)

    # Create and train model
    logger.info("starting model tuning", tuned_model_id=new_model_id)
    operation = genai.create_tuned_model(
        id=new_model_id,
        source_model=settings.tuning_source_model,
        training_data=training_dataset,
        epoch_count=settings.tuning_epoch_count,
        batch_size=settings.tuning_batch_size,
        learning_rate=settings.tuning_learning_rate,
    )

    # Monitor training progress
    logger.info("tuning model (takes a few mins)", tuned_model_id=new_model_id)
    for _ in operation.wait_bar():
        pass

    tuned_model = operation.result()
    logger.info("tuning complete", tuned_model=tuned_model)

    # Generate and save loss plot
    snapshots = pd.DataFrame(tuned_model.tuning_task.snapshots)
    plot_path = save_loss_plot(snapshots, new_model_id)
    logger.info("saved mean_loss plot", save_fig_path=str(plot_path))


if __name__ == "__main__":
    start()
