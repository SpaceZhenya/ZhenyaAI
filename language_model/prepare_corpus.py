from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import json
import random
import re


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_IMPORTED_DIR = PROJECT_ROOT / "data" / "raw" / "imported"
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

TRAIN_PATH = SPLITS_DIR / "train.txt"
VALIDATION_PATH = SPLITS_DIR / "validation.txt"
TEST_PATH = SPLITS_DIR / "test.txt"
STATS_PATH = SPLITS_DIR / "corpus_stats.json"

MIN_CHARACTERS = 200

TRAIN_FRACTION = 0.90
VALIDATION_FRACTION = 0.05
TEST_FRACTION = 0.05

SEED = 42


@dataclass
class CorpusStats:
    found_files: int
    accepted_documents: int
    duplicate_documents: int
    rejected_documents: int
    total_characters: int
    train_characters: int
    validation_characters: int
    test_characters: int


def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    lines = []

    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()

        if not line:
            continue

        if line.startswith(("http://", "https://")):
            continue

        lines.append(line)

    return "\n".join(lines).strip()


def load_documents() -> tuple[list[str], CorpusStats]:
    RAW_IMPORTED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(
        RAW_IMPORTED_DIR.rglob("*.txt")
    )

    if not files:
        raise FileNotFoundError(
            "Нет файлов для подготовки корпуса.\n"
            "Добавь TXT-файлы в:\n"
            "data/raw/imported/"
        )

    documents: list[str] = []
    hashes: set[str] = set()

    duplicates = 0
    rejected = 0

    for path in files:
        raw_text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        text = normalize_text(raw_text)

        if len(text) < MIN_CHARACTERS:
            rejected += 1
            print(
                f"Пропущен короткий файл: {path.name}"
            )
            continue

        document_hash = sha256(
            text.encode("utf-8")
        ).hexdigest()

        if document_hash in hashes:
            duplicates += 1
            print(
                f"Пропущен дубликат: {path.name}"
            )
            continue

        hashes.add(document_hash)
        documents.append(text)

        print(
            f"Добавлен файл: {path.name} "
            f"({len(text):,} символов)"
        )

    stats = CorpusStats(
        found_files=len(files),
        accepted_documents=len(documents),
        duplicate_documents=duplicates,
        rejected_documents=rejected,
        total_characters=sum(
            len(document)
            for document in documents
        ),
        train_characters=0,
        validation_characters=0,
        test_characters=0,
    )

    if not documents:
        raise ValueError(
            "Нет подходящих документов после очистки."
        )

    return documents, stats


def split_documents(
    documents: list[str],
) -> tuple[list[str], list[str], list[str]]:
    if len(documents) < 3:
        raise ValueError(
            "Нужно минимум 3 документа: "
            "для train, validation и test."
        )

    random.Random(SEED).shuffle(documents)

    total = len(documents)

    train_end = max(
        1,
        int(total * TRAIN_FRACTION),
    )

    validation_end = max(
        train_end + 1,
        int(total * (TRAIN_FRACTION + VALIDATION_FRACTION)),
    )

    validation_end = min(
        validation_end,
        total - 1,
    )

    train = documents[:train_end]
    validation = documents[
        train_end:validation_end
    ]
    test = documents[validation_end:]

    if not validation or not test:
        raise ValueError(
            "Недостаточно документов для разделения. "
            "Добавь больше отдельных TXT-файлов."
        )

    return train, validation, test


def save_split(
    path: Path,
    documents: list[str],
) -> int:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    text = "\n\n<eos>\n\n".join(documents)
    path.write_text(
        text,
        encoding="utf-8",
    )

    return len(text)


def save_stats(stats: CorpusStats) -> None:
    STATS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with STATS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            asdict(stats),
            file,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    print("Подготовка корпуса ZhenyaAI...\n")

    documents, stats = load_documents()

    train, validation, test = split_documents(
        documents
    )

    train_characters = save_split(
        TRAIN_PATH,
        train,
    )

    validation_characters = save_split(
        VALIDATION_PATH,
        validation,
    )

    test_characters = save_split(
        TEST_PATH,
        test,
    )

    stats.train_characters = train_characters
    stats.validation_characters = validation_characters
    stats.test_characters = test_characters

    save_stats(stats)

    print("\nКорпус подготовлен.")
    print(
        f"Документов принято: "
        f"{stats.accepted_documents}"
    )
    print(
        f"Всего символов: "
        f"{stats.total_characters:,}"
    )
    print(
        f"Train: {train_characters:,} символов"
    )
    print(
        f"Validation: "
        f"{validation_characters:,} символов"
    )
    print(
        f"Test: {test_characters:,} символов"
    )
    print(f"\nФайлы сохранены в: {SPLITS_DIR}")


if __name__ == "__main__":
    main()
