from pathlib import Path

import structlog
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = structlog.get_logger(__name__)


class Settings(BaseSettings):
    """
    Application settings model that provides configuration for all components.
    """

    # API key for accessing Google's Gemini AI service
    gemini_api_key: str = ""
    # Name of the new tuned model
    tuned_model_name: str = ""
    # Base model to tune upon
    tuning_source_model: str = "models/gemini-1.5-flash-001-tuning"
    # Tuning dataset path
    tuning_dataset_path: Path = (
        Path(__file__).parent.parent / "data" / "training_data.json"
    )
    # Number of epochs to tune for
    tuning_epoch_count: int = 30
    # Batch size
    tuning_batch_size: int = 4
    # Learning rate
    tuning_learning_rate: float = 0.001
    # X keys and token (optional)
    x_bearer_token: str = ""
    x_api_key: str = ""
    x_api_key_secret: str = ""
    x_access_token: str = ""
    x_access_token_secret: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


# Create a global settings instance
settings = Settings()
logger.debug("settings", settings=settings.model_dump())
