from pathlib import Path
import random

import torch

from core.model import create_small_model
from core.tokenizer import CharacterTokenizer


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "core" / "data" / "train.txt"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
MODEL_PATH = CHECKPOINT_DIR / "zhenyaai.pt"
TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer.json"

BATCH_SIZE = 16
BLOCK_SIZE = 128
STEPS = 2000
LEARNING_RATE = 3e-4
PRINT_EVERY = 100
SEED = 42


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def create_batch(
    token_ids: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(token_ids) - block_size - 1

    if max_start < 1:
        raise ValueError(
            "Датасет слишком маленький. Добавь больше текста в core/data/train.txt."
        )

    starts = torch.randint(0, max_start, (batch_size,))
    inputs = torch.stack(
        [token_ids[start:start + block_size] for start in starts]
    )
    targets = torch.stack(
        [token_ids[start + 1:start + block_size + 1] for start in starts]
    )

    return inputs.to(device), targets.to(device)


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Не найден датасет: {DATA_PATH}. "
            "Создай файл core/data/train.txt."
        )

    text = DATA_PATH.read_text(encoding="utf-8").strip()

    if len(text) < 300:
        raise ValueError(
            "Датасет слишком маленький. Добавь хотя бы 300 символов текста."
        )

    tokenizer = CharacterTokenizer(text)
    token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    device = get_device()

    print(f"Устройство: {device}")
    print(f"Символов в датасете: {len(text):,}")
    print(f"Размер словаря: {tokenizer.vocab_size}")
    print("Создаём ZhenyaTransformer...")

    model = create_small_model(tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print(f"Параметров в модели: {model.parameter_count():,}")
    print("Начинаем обучение...")

    model.train()

    for step in range(1, STEPS + 1):
        inputs, targets = create_batch(
            token_ids=token_ids,
            batch_size=BATCH_SIZE,
            block_size=BLOCK_SIZE,
            device=device,
        )

        _, loss = model(inputs, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % PRINT_EVERY == 0:
            print(f"Шаг {step}/{STEPS} | loss: {loss.item():.4f}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": tokenizer.vocab_size,
            "block_size": model.block_size,
            "training_steps": STEPS,
        },
        MODEL_PATH,
    )

    tokenizer.save(str(TOKENIZER_PATH))

    print("\nОбучение завершено.")
    print(f"Веса модели сохранены: {MODEL_PATH}")
    print(f"Токенизатор сохранён: {TOKENIZER_PATH}")


if __name__ == "__main__":
    main()
