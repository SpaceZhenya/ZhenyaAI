from __future__ import annotations

from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tokenizers import Tokenizer

from language_model.config import ModelConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

TOKENIZER_PATH = PROCESSED_DATA_DIR / "tokenizer.json"
TOKENIZER_INFO_PATH = PROCESSED_DATA_DIR / "tokenizer_info.json"

TRAIN_TOKENS_PATH = PROCESSED_DATA_DIR / "train_tokens.pt"
VALID_TOKENS_PATH = PROCESSED_DATA_DIR / "valid_tokens.pt"

VALIDATION_FRACTION = 0.2
SPLIT_SEED = 42


def read_corpus() -> str:
    files = sorted(RAW_DATA_DIR.rglob("*.txt"))

    if not files:
        raise FileNotFoundError(
            "Не найдены TXT-файлы в data/raw."
        )

    parts: list[str] = []

    for path in files:
        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        ).strip()

        if text:
            parts.append(text)

            print(
                f"Прочитан файл: {path.name} "
                f"({len(text):,} символов)"
            )

    corpus = "\n\n".join(parts)

    if not corpus:
        raise ValueError("Корпус пустой.")

    return corpus


def load_tokenizer() -> Tokenizer:
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            "Токенизатор не найден.\n"
            "Сначала запусти:\n"
            "python -m language_model.tokenizer"
        )

    return Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )


def get_tokenizer_vocab_size() -> int:
    tokenizer = load_tokenizer()
    return tokenizer.get_vocab_size()


def prepare_token_data() -> tuple[torch.Tensor, torch.Tensor]:
    corpus = read_corpus()
    tokenizer = load_tokenizer()

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    if bos_id is None or eos_id is None:
        raise ValueError(
            "В токенизаторе нет <bos> или <eos>."
        )

    encoded = tokenizer.encode(corpus)

    token_ids = [
        bos_id,
        *encoded.ids,
        eos_id,
    ]

    if len(token_ids) < 20:
        raise ValueError(
            "После токенизации слишком мало токенов."
        )

    random.seed(SPLIT_SEED)

    split_index = int(
        len(token_ids) * (1.0 - VALIDATION_FRACTION)
    )

    split_index = max(
        10,
        min(split_index, len(token_ids) - 10),
    )

    train_ids = token_ids[:split_index]
    valid_ids = token_ids[split_index:]

    train_tokens = torch.tensor(
        train_ids,
        dtype=torch.long,
    )

    valid_tokens = torch.tensor(
        valid_ids,
        dtype=torch.long,
    )

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(train_tokens, TRAIN_TOKENS_PATH)
    torch.save(valid_tokens, VALID_TOKENS_PATH)

    print("\nПодготовка данных завершена.")
    print(f"Всего токенов: {len(token_ids):,}")
    print(f"Токенов для обучения: {len(train_tokens):,}")
    print(f"Токенов для проверки: {len(valid_tokens):,}")

    return train_tokens, valid_tokens


class NextTokenDataset(Dataset):
    def __init__(
        self,
        token_ids: torch.Tensor,
        context_length: int,
    ) -> None:
        if token_ids.dim() != 1:
            raise ValueError(
                "token_ids должен быть одномерным тензором."
            )

        if len(token_ids) <= context_length:
            raise ValueError(
                "Недостаточно токенов для выбранного context_length."
            )

        self.token_ids = token_ids
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.token_ids) - self.context_length

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start = index
        end = start + self.context_length

        input_ids = self.token_ids[start:end]
        targets = self.token_ids[start + 1:end + 1]

        return input_ids, targets


def choose_context_length(
    train_tokens: torch.Tensor,
    valid_tokens: torch.Tensor,
    requested_length: int,
) -> int:
    """
    Выбирает длину контекста, подходящую обоим наборам.
    """

    maximum_length = min(
        len(train_tokens) - 1,
        len(valid_tokens) - 1,
    )

    if maximum_length < 8:
        raise ValueError(
            "Validation-набор слишком маленький. "
            "Добавь больше текста в data/raw."
        )

    return max(
        8,
        min(requested_length, maximum_length),
    )


def create_data_loader(
    token_ids: torch.Tensor,
    config: ModelConfig,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = NextTokenDataset(
        token_ids=token_ids,
        context_length=config.context_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
    )


def main() -> None:
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

    train_loader = create_data_loader(
        token_ids=train_tokens,
        config=config,
        batch_size=2,
        shuffle=False,
    )

    valid_loader = create_data_loader(
        token_ids=valid_tokens,
        config=config,
        batch_size=2,
        shuffle=False,
    )

    train_input_ids, train_targets = next(iter(train_loader))
    valid_input_ids, valid_targets = next(iter(valid_loader))

    print("\nТест DataLoader:")
    print(f"Размер словаря: {config.vocab_size}")
    print(f"Context length: {config.context_length}")
    print(
        f"Train input_ids: "
        f"{tuple(train_input_ids.shape)}"
    )
    print(
        f"Train targets: "
        f"{tuple(train_targets.shape)}"
    )
    print(
        f"Valid input_ids: "
        f"{tuple(valid_input_ids.shape)}"
    )
    print(
        f"Valid targets: "
        f"{tuple(valid_targets.shape)}"
    )


if __name__ == "__main__":
    main()
