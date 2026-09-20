from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer

from language_model.config import ModelConfig
from language_model.model import ZhenyaLanguageModel


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
            "Не найден токенизатор:\n"
            f"{TOKENIZER_PATH}\n\n"
            "Сначала запусти:\n"
            "python -m language_model.tokenizer"
        )

    return Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )


def load_model() -> ZhenyaLanguageModel:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "Не найден checkpoint:\n"
            f"{CHECKPOINT_PATH}\n\n"
            "Сначала запусти:\n"
            "python -m language_model.train"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model_config = ModelConfig(
        **checkpoint["model_config"]
    )

    model = ZhenyaLanguageModel(
        model_config
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    train_loss = checkpoint.get(
        "train_loss",
        float("nan"),
    )

    validation_loss = checkpoint.get(
        "validation_loss",
        float("nan"),
    )

    print("Модель загружена.")
    print(
        f"Шаг обучения: "
        f"{checkpoint.get('step', '?')}"
    )
    print(
        f"Train loss: {train_loss:.4f}"
    )
    print(
        f"Validation loss: "
        f"{validation_loss:.4f}"
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
        raise ValueError(
            "Промпт не может быть пустым."
        )

    if max_new_tokens < 1:
        raise ValueError(
            "max_new_tokens должен быть больше нуля."
        )

    if temperature <= 0:
        raise ValueError(
            "temperature должен быть больше нуля."
        )

    encoded = tokenizer.encode(prompt)

    if not encoded.ids:
        raise ValueError(
            "Токенизатор не получил токены."
        )

    input_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    generated_text = tokenizer.decode(
        generated_ids[0].tolist(),
        skip_special_tokens=True,
    )

    return generated_text


def main() -> None:
    tokenizer = load_tokenizer()
    model = load_model()

    print("\nZhenyaAI — тест собственной модели.")
    print(
        "Напиши начало текста, и модель "
        "попробует его продолжить."
    )
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
                max_new_tokens=80,
                temperature=0.7,
                top_k=20,
            )

            print(f"\nZhenyaAI: {answer}\n")

        except Exception as error:
            print(
                f"\nОшибка: {error}\n"
            )


if __name__ == "__main__":
    main()
