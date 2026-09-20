from __future__ import annotations

from pathlib import Path
import random

import torch

from language_model.config import (
    CHECKPOINTS_DIR,
    ModelConfig,
    TrainingConfig,
    create_project_directories,
)
from language_model.dataset import (
    create_data_loader,
    prepare_token_data,
)
from language_model.model import ZhenyaLanguageModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BEST_CHECKPOINT_PATH = CHECKPOINTS_DIR / "best_model.pt"
LAST_CHECKPOINT_PATH = CHECKPOINTS_DIR / "last_model.pt"

VOCAB_SIZE = 8_000
BATCH_SIZE = 2
MAX_STEPS = 500
LEARNING_RATE = 3e-4
EVAL_EVERY_STEPS = 50
SAVE_EVERY_STEPS = 100
SEED = 42


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def evaluate_loss(
    model: ZhenyaLanguageModel,
    loader,
    device: torch.device,
    batches: int = 10,
) -> float:
    model.eval()

    losses: list[float] = []

    with torch.no_grad():
        for batch_index, (input_ids, targets) in enumerate(loader):
            if batch_index >= batches:
                break

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            _, loss = model(
                input_ids=input_ids,
                targets=targets,
            )

            if loss is not None:
                losses.append(loss.item())

    model.train()

    if not losses:
        return float("inf")

    return sum(losses) / len(losses)


def save_checkpoint(
    path: Path,
    model: ZhenyaLanguageModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    train_loss: float,
    validation_loss: float,
    config: ModelConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "step": step,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "model_config": config.__dict__,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)

    create_project_directories()

    print("Подготавливаем токены...")

    train_tokens, valid_tokens = prepare_token_data()

    config = ModelConfig(
        vocab_size=VOCAB_SIZE,
    )

    training_config = TrainingConfig(
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_steps=MAX_STEPS,
        eval_every_steps=EVAL_EVERY_STEPS,
        save_every_steps=SAVE_EVERY_STEPS,
        seed=SEED,
    )

    device = get_device()

    print(f"\nУстройство: {device}")
    print(f"Размер словаря: {config.vocab_size}")
    print(f"Длина контекста: {config.context_length}")

    train_loader = create_data_loader(
        token_ids=train_tokens,
        config=config,
        batch_size=training_config.batch_size,
        shuffle=True,
    )

    valid_loader = create_data_loader(
        token_ids=valid_tokens,
        config=config,
        batch_size=training_config.batch_size,
        shuffle=False,
    )

    model = ZhenyaLanguageModel(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    print(
        f"Параметров модели: "
        f"{model.parameter_count():,}"
    )
    print(f"Максимальное количество шагов: {MAX_STEPS}")
    print("Начинаем обучение...\n")

    train_iterator = iter(train_loader)
    best_validation_loss = float("inf")

    for step in range(1, training_config.max_steps + 1):
        try:
            input_ids, targets = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            input_ids, targets = next(train_iterator)

        input_ids = input_ids.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        _, loss = model(
            input_ids=input_ids,
            targets=targets,
        )

        if loss is None:
            raise RuntimeError("Модель не вернула loss.")

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        if step == 1 or step % EVAL_EVERY_STEPS == 0:
            validation_loss = evaluate_loss(
                model=model,
                loader=valid_loader,
                device=device,
            )

            print(
                f"Шаг {step}/{MAX_STEPS} | "
                f"train loss: {loss.item():.4f} | "
                f"valid loss: {validation_loss:.4f}"
            )

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss

                save_checkpoint(
                    path=BEST_CHECKPOINT_PATH,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    train_loss=loss.item(),
                    validation_loss=validation_loss,
                    config=config,
                )

                print(
                    f"Лучшая модель сохранена: "
                    f"{BEST_CHECKPOINT_PATH}"
                )

        if step % SAVE_EVERY_STEPS == 0:
            save_checkpoint(
                path=LAST_CHECKPOINT_PATH,
                model=model,
                optimizer=optimizer,
                step=step,
                train_loss=loss.item(),
                validation_loss=validation_loss
                if "validation_loss" in locals()
                else float("inf"),
                config=config,
            )

            print(
                f"Checkpoint сохранён: "
                f"{LAST_CHECKPOINT_PATH}"
            )

    print("\nОбучение завершено.")
    print(f"Лучшая модель: {BEST_CHECKPOINT_PATH}")
    print(f"Последняя модель: {LAST_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
