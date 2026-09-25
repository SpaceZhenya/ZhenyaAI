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
        grouped_parameters[group_name]
