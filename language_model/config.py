from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DIALOGS_DATA_DIR = DATA_DIR / "dialogs"

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"


@dataclass(frozen=True)
class ModelConfig:
    """
    Настройки первой собственной языковой модели ZhenyaAI.
    """

    vocab_size: int = 8_000
    context_length: int = 256

    embedding_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1

    def validate(self) -> None:
        if self.vocab_size < 256:
            raise ValueError("vocab_size должен быть не меньше 256.")

        if self.context_length < 16:
            raise ValueError("context_length должен быть не меньше 16.")

        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                "embedding_dim должен делиться на num_heads без остатка."
            )

        if self.num_layers < 1:
            raise ValueError("num_layers должен быть не меньше 1.")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout должен быть от 0.0 до 0.99.")


@dataclass(frozen=True)
class TrainingConfig:
    """
    Настройки обучения модели.
    """

    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1

    max_steps: int = 20_000
    warmup_steps: int = 500

    eval_every_steps: int = 250
    save_every_steps: int = 1_000

    seed: int = 42


def create_project_directories() -> None:
    """
    Создаёт папки, которые нужны для данных, весов и логов.
    """

    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        DIALOGS_DATA_DIR,
        CHECKPOINTS_DIR,
        LOGS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def save_model_config(
    config: ModelConfig,
    path: Path,
) -> None:
    """
    Сохраняет конфигурацию рядом с весами модели.
    """

    config.validate()

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            asdict(config),
            file,
            ensure_ascii=False,
            indent=2,
        )


def load_model_config(path: Path) -> ModelConfig:
    """
    Загружает конфигурацию, с которой модель была создана.
    """

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    config = ModelConfig(**data)
    config.validate()

    return config


if __name__ == "__main__":
    model_config = ModelConfig()
    training_config = TrainingConfig()

    model_config.validate()
    create_project_directories()

    print("Настройки ZhenyaAI готовы.")
    print(f"Размер словаря: {model_config.vocab_size}")
    print(f"Длина контекста: {model_config.context_length}")
    print(f"Размер embedding: {model_config.embedding_dim}")
    print(f"Transformer-слоёв: {model_config.num_layers}")
    print(f"Attention heads: {model_config.num_heads}")
    print(f"Шагов обучения: {training_config.max_steps}")
