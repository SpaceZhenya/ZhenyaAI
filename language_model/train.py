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
    choose_context_length,
    create_data_loader,
    get_tokenizer_vocab_size,
    prepare_token_data,
)
from language_model.model import ZhenyaLanguageModel


BEST_CHECKPOINT_PATH = CHECKPOINTS_DIR / "best_model.pt"
LAST_CHECKPOINT_PATH = CHECKPOINTS_DIR / "last_model.pt"

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

    losses = []

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
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "step": step,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "model_config": {
                "vocab_size": config.vocab_size,
                "context_length": config.context_length,
                "embedding_dim": config.embedding_dim,
                "num_layers": config.num_layers,
                "num_heads": config.num_heads,
                "dropout": config.dropout,
            },
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)

    create_project_directories()

    print("Подготавливаем токены...\n")

    train_tokens, valid_tokens = prepare_token_data()

    vocab_size = get_tokenizer_vocab_size()

    context_length = choose_context_length(
        train_tokens=train_tokens,
        valid_tokens=valid_tokens,
        requested_length=256,
    )

    config = ModelConfig(
        vocab_size=vocab_size,
        context_length=context_length,
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

    print(f"Устройство: {device}")
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
    print(
        f"Максимальное количество шагов: "
        f"{training_config.max_steps}"
    )
    print("\nНачинаем обучение...\n")

    train_iterator = iter(train_loader)
    best_validation_loss = float("inf")
    last_validation_loss = float("inf")

    for step in range(
        1,
        training_config.max_steps + 1,
    ):
        try:
            input_ids, targets = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            input_ids, targets = next(train_iterator)

        input_ids = input_ids.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(
            set_to_none=True,
        )

        _, loss = model(
            input_ids=input_ids,
            targets=targets,
        )

        if loss is None:
            raise RuntimeError(
                "Модель не вернула значение loss."
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        should_evaluate = (
            step == 1
            or step % training_config.eval_every_steps == 0
        )

        if should_evaluate:
            last_validation_loss = evaluate_loss(
                model=model,
                loader=valid_loader,
                device=device,
            )

            print(
                f"Шаг {step}/{training_config.max_steps} | "
                f"train loss: {loss.item():.4f} | "
                f"valid loss: {last_validation_loss:.4f}"
            )

            if last_validation_loss < best_validation_loss:
                best_validation_loss = last_validation_loss

                save_checkpoint(
                    path=BEST_CHECKPOINT_PATH,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    train_loss=loss.item(),
                    validation_loss=last_validation_loss,
                    config=config,
                )

                print(
                    "Лучшая модель сохранена: "
                    f"{BEST_CHECKPOINT_PATH}"
                )

        if (
            step % training_config.save_every_steps == 0
            or step == training_config.max_steps
        ):
            save_checkpoint(
                path=LAST_CHECKPOINT_PATH,
                model=model,
                optimizer=optimizer,
                step=step,
                train_loss=loss.item(),
                validation_loss=last_validation_loss,
                config=config,
            )

            print(
                "Checkpoint сохранён: "
                f"{LAST_CHECKPOINT_PATH}"
            )

    print("\nОбучение завершено.")
    print(f"Лучшая модель: {BEST_CHECKPOINT_PATH}")
    print(f"Последняя модель: {LAST_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
