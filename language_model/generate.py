from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer

from language_model.model import ZhenyaLanguageModel
from language_model.config import ModelConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tokenizer.json"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "best_model.pt"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def load_tokenizer() -> Tokenizer:
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            "Токенизатор не найден: "
            f"{TOKENIZER_PATH}"
        )

    return Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )


def load_model() -> ZhenyaLanguageModel:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "Checkpoint не найден: "
            f"{CHECKPOINT_PATH}\n"
            "Сначала запусти обучение:\n"
            "python -m language_model.train"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    config = ModelConfig(
        **checkpoint["model_config"]
    )

    model = ZhenyaLanguageModel(config)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    print(
        f"Модель загружена. "
        f"Шаг обучения: {checkpoint['step']}"
    )
    print(
        f"Validation loss: "
        f"{checkpoint['validation_loss']:.4f}"
    )

    return model


def generate_text(
    model: ZhenyaLanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    top_k: int = 20,
) -> str:
    prompt = prompt.strip()

    if not prompt:
        raise ValueError("Текст не может быть пустым.")

    encoded = tokenizer.encode(prompt)

    if not encoded.ids:
        raise ValueError(
            "Не удалось преобразовать текст в токены."
        )

    input_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=DEVICE,
    )

    generated_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    return tokenizer.decode(
        generated_ids[0].tolist(),
        skip_special_tokens=True,
    )


def main() -> None:
    tokenizer = load_tokenizer()
    model = load_model()

    print("\nZhenyaAI — проверка генерации текста.")
    print("Напиши фразу, а модель попробует её продолжить.")
    print("Для выхода введи: выход\n")

    while True:
        prompt = input("Ты: ").strip()

        if prompt.lower() in {
            "выход",
            "exit",
            "quit",
        }:
            print("ZhenyaAI: До встречи!")
            break

        if not prompt:
            continue

        try:
            answer = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
            )

            print(f"ZhenyaAI: {answer}\n")

        except Exception as error:
            print(f"Ошибка: {error}\n")


if __name__ == "__main__":
    main()
