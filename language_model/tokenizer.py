from __future__ import annotations

from pathlib import Path
import json

from tokenizers import Tokenizer
from tokenizers import models
from tokenizers import pre_tokenizers
from tokenizers import trainers
from tokenizers import decoders


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
TOKENIZER_PATH = PROCESSED_DATA_DIR / "tokenizer.json"

VOCAB_SIZE = 8_000

SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<user>",
    "<assistant>",
    "<system>",
    "<tool>",
]


def find_training_files() -> list[str]:
    """
    Находит все TXT-файлы в data/raw и вложенных папках.
    """

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(
        RAW_DATA_DIR.rglob("*.txt")
    )

    if not files:
        raise FileNotFoundError(
            "В папке data/raw нет TXT-файлов.\n"
            "Добавь файл data/raw/zhenyaai_corpus.txt."
        )

    return [str(file) for file in files]


def build_tokenizer() -> Tokenizer:
    """
    Создаёт BPE-токенизатор с нуля.
    """

    tokenizer = Tokenizer(
        models.BPE(
            unk_token="<unk>",
        )
    )

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
    )

    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=1,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    training_files = find_training_files()

    print("Файлы для обучения токенизатора:")

    for file_path in training_files:
        print(f"- {file_path}")

    tokenizer.train(
        files=training_files,
        trainer=trainer,
    )

    return tokenizer


def save_tokenizer(tokenizer: Tokenizer) -> None:
    """
    Сохраняет токенизатор в JSON.
    """

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(TOKENIZER_PATH))

    print(f"\nТокенизатор сохранён: {TOKENIZER_PATH}")
    print(f"Размер словаря: {tokenizer.get_vocab_size()}")


def test_tokenizer(tokenizer: Tokenizer) -> None:
    """
    Проверяет кодирование и обратную расшифровку текста.
    """

    sample = (
        "Привет! Я ZhenyaAI. "
        "Я учусь работать с текстом."
    )

    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded.ids)

    print("\nТест токенизатора:")
    print(f"Исходный текст: {sample}")
    print(f"ID токенов: {encoded.ids}")
    print(f"Количество токенов: {len(encoded.ids)}")
    print(f"Расшифровка: {decoded}")


def save_tokenizer_info(tokenizer: Tokenizer) -> None:
    """
    Сохраняет краткую информацию о токенизаторе.
    """

    info_path = PROCESSED_DATA_DIR / "tokenizer_info.json"

    special_token_ids = {
        token: tokenizer.token_to_id(token)
        for token in SPECIAL_TOKENS
    }

    info = {
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": special_token_ids,
        "tokenizer_type": "BPE",
        "source_directory": str(RAW_DATA_DIR),
    }

    with info_path.open("w", encoding="utf-8") as file:
        json.dump(
            info,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Информация сохранена: {info_path}")


def main() -> None:
    tokenizer = build_tokenizer()
    save_tokenizer(tokenizer)
    save_tokenizer_info(tokenizer)
    test_tokenizer(tokenizer)


if __name__ == "__main__":
    main()
