from __future__ import annotations

import torch

from language_model.config import ModelConfig
from language_model.dataset import (
    choose_context_length,
    create_data_loader,
    get_tokenizer_vocab_size,
    prepare_token_data,
)
from language_model.model import ZhenyaLanguageModel


SEED = 42
BATCH_SIZE = 2
LEARNING_RATE = 3e-4


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def main() -> None:
    torch.manual_seed(SEED)

    device = get_device()

    print("ПРОВЕРКА ИЗМЕНЕНИЯ ВЕСОВ")
    print("-" * 55)
    print(f"Устройство: {device}\n")

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

    loader = create_data_loader(
        token_ids=train_tokens,
        config=config,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = ZhenyaLanguageModel(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    input_ids, targets = next(iter(loader))

    input_ids = input_ids.to(device)
    targets = targets.to(device)

    watched_name = ""
    watched_before = None

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            watched_name = name
            watched_before = parameter.detach().clone()
            break

    if watched_before is None:
        raise RuntimeError(
            "Не найден параметр, который можно обучать."
        )

    model.train()

    optimizer.zero_grad(set_to_none=True)

    _, loss_before = model(
        input_ids=input_ids,
        targets=targets,
    )

    if loss_before is None:
        raise RuntimeError(
            "Модель не вернула loss."
        )

    loss_before.backward()

    watched_gradient = dict(
        model.named_parameters()
    )[watched_name].grad

    if watched_gradient is None:
        raise RuntimeError(
            "Градиент проверяемого параметра не найден."
        )

    gradient_norm = watched_gradient.norm().item()

    optimizer.step()

    watched_after = dict(
        model.named_parameters()
    )[watched_name].detach().clone()

    difference = torch.abs(
        watched_after - watched_before
    )

    changed_values = int(
        torch.count_nonzero(difference).item()
    )

    total_values = difference.numel()

    maximum_change = difference.max().item()
    average_change = difference.mean().item()

    print("РЕЗУЛЬТАТ ОДНОГО ШАГА ОБУЧЕНИЯ")
    print("-" * 55)
    print(f"Проверяемый параметр: {watched_name}")
    print(f"Loss до обновления: {loss_before.item():.6f}")
    print(f"Норма градиента: {gradient_norm:.8f}")
    print(
        "Изменившихся чисел: "
        f"{changed_values}/{total_values}"
    )
    print(
        "Максимальное изменение: "
        f"{maximum_change:.12f}"
    )
    print(
        "Среднее изменение: "
        f"{average_change:.12f}"
    )

    if changed_values == 0:
        raise RuntimeError(
            "Весы не изменились после optimizer.step()."
        )

    print("\nПРОВЕРКА ПРОЙДЕНА")
    print(
        "Loss создал градиенты, "
        "а AdamW обновил веса нейросети."
    )


if __name__ == "__main__":
    main()
