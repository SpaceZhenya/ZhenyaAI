from __future__ import annotations

from collections import defaultdict

import torch

from language_model.config import ModelConfig
from language_model.dataset import (
    choose_context_length,
    get_tokenizer_vocab_size,
    prepare_token_data,
)
from language_model.model import ZhenyaLanguageModel


def format_number(number: int) -> str:
    return f"{number:,}".replace(",", " ")


def format_memory(byte_count: int) -> str:
    units = ["байт", "КБ", "МБ", "ГБ"]
    value = float(byte_count)

    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"

        value /= 1024

    return f"{value:.2f} ГБ"


def print_config(config: ModelConfig) -> None:
    print("КОНФИГУРАЦИЯ МОДЕЛИ")
    print("-" * 55)

    values = [
        ("Размер словаря", config.vocab_size),
        ("Длина контекста", config.context_length),
        ("Размер embedding", config.embedding_dim),
        ("Количество Transformer-блоков", config.num_layers),
        ("Количество attention-голов", config.num_heads),
        ("Dropout", config.dropout),
    ]

    for name, value in values:
        print(f"{name}: {value}")

    print()


def inspect_parameters(
    model: ZhenyaLanguageModel,
) -> None:
    print("ОБУЧАЕМЫЕ ВЕСА И СВЯЗИ")
    print("-" * 55)

    total_parameters = 0
    trainable_parameters = 0
    total_bytes = 0

    grouped_parameters: dict[str, int] = defaultdict(int)

    for name, parameter in model.named_parameters():
        count = parameter.numel()
        byte_count = (
            count * parameter.element_size()
        )

        total_parameters += count
        total_bytes += byte_count

        if parameter.requires_grad:
            trainable_parameters += count

        group_name = name.split(".")[0]
        grouped_parameters[group_name] += count

        print(
            f"{name}\n"
            f"  форма: {tuple(parameter.shape)}\n"
            f"  чисел-весов: {format_number(count)}\n"
            f"  обучается: {parameter.requires_grad}\n"
        )

    print("ИТОГ")
    print("-" * 55)
    print(
        "Всего чисел в весах: "
        f"{format_number(total_parameters)}"
    )
    print(
        "Обучаемых чисел: "
        f"{format_number(trainable_parameters)}"
    )
    print(
        "Память для весов: "
        f"{format_memory(total_bytes)}"
    )

    print("\nВЕСА ПО ОСНОВНЫМ ЧАСТЯМ")
    print("-" * 55)

    for group_name, count in sorted(
        grouped_parameters.items()
    ):
        print(
            f"{group_name}: "
            f"{format_number(count)}"
        )

    print()


def inspect_modules(
    model: ZhenyaLanguageModel,
) -> None:
    print("СЛОИ НЕЙРОСЕТИ")
    print("-" * 55)

    module_count = 0

    for name, module in model.named_modules():
        if not name:
            continue

        module_count += 1

        print(
            f"{name}: "
            f"{module.__class__.__name__}"
        )

    print(
        f"\nВсего модулей: "
        f"{module_count}"
    )
    print()


def inspect_forward_pass(
    model: ZhenyaLanguageModel,
    config: ModelConfig,
) -> None:
    print("ПРОХОД ТОКЕНОВ ЧЕРЕЗ НЕЙРОСЕТЬ")
    print("-" * 55)

    device = next(model.parameters()).device

    example_input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(1, min(8, config.context_length)),
        device=device,
    )

    print(
        "Пример токенов на входе: "
        f"{example_input_ids.tolist()}"
    )
    print(
        "Форма входа: "
        f"{tuple(example_input_ids.shape)}"
    )

    with torch.no_grad():
        logits, loss = model(
            input_ids=example_input_ids
        )

    print(
        "Форма выхода logits: "
        f"{tuple(logits.shape)}"
    )
    print(
        "Форма означает: "
        "(batch, токены, варианты следующего токена)"
    )
    print(
        "Loss без targets: "
        f"{loss}"
    )

    next_token_logits = logits[0, -1]

    probabilities = torch.softmax(
        next_token_logits,
        dim=-1,
    )

    top_probabilities, top_token_ids = torch.topk(
        probabilities,
        k=min(5, config.vocab_size),
    )

    print("\n5 вариантов следующего токена:")
    for token_id, probability in zip(
        top_token_ids.tolist(),
        top_probabilities.tolist(),
    ):
        print(
            f"Токен {token_id}: "
            f"{probability * 100:.2f}%"
        )

    print()


def main() -> None:
    print("\nИНСПЕКТОР НЕЙРОСЕТИ ZhenyaAI\n")

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

    model = ZhenyaLanguageModel(config)

    print_config(config)
    inspect_parameters(model)
    inspect_modules(model)
    inspect_forward_pass(model, config)

    print("ГОТОВО")
    print("-" * 55)
    print(
        "Нейросеть создана, но ещё не загружает "
        "checkpoint. Поэтому веса сейчас случайные."
    )
    print(
        "На следующем шаге сравним веса до и после "
        "настоящего обучения."
    )


if __name__ == "__main__":
    main()
