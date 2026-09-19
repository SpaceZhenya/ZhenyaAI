from pathlib import Path

import torch

from core.model import create_small_model
from core.tokenizer import CharacterTokenizer


ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
MODEL_PATH = CHECKPOINT_DIR / "zhenyaai.pt"
TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer.json"

NEW_TOKENS = 280
TEMPERATURE = 0.45


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def load_model() -> tuple[torch.nn.Module, CharacterTokenizer, str]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Веса модели не найдены.\n"
            "Сначала выполни: python -m core.train"
        )

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            "Токенизатор не найден.\n"
            "Сначала выполни: python -m core.train"
        )

    device = get_device()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    tokenizer = CharacterTokenizer.load(str(TOKENIZER_PATH))

    model = create_small_model(
        vocab_size=checkpoint["vocab_size"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, tokenizer, device


def make_prompt(user_text: str) -> str:
    return f"Пользователь: {user_text}\nZhenyaAI:"


def generate_answer(
    model: torch.nn.Module,
    tokenizer: CharacterTokenizer,
    device: str,
    prompt: str,
) -> str:
    token_ids = tokenizer.encode(prompt)

    input_tokens = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )

    generated_tokens = model.generate(
        tokens=input_tokens,
        new_tokens=NEW_TOKENS,
        temperature=TEMPERATURE,
    )

    text = tokenizer.decode(generated_tokens[0].tolist())

    if "ZhenyaAI:" in text:
        answer = text.rsplit("ZhenyaAI:", maxsplit=1)[-1]
    else:
        answer = text

    answer = answer.split("Пользователь:")[0].strip()

    return answer


def main() -> None:
    model, tokenizer, device = load_model()

    print("ZhenyaAI запущен.")
    print("Напиши вопрос на русском.")
    print("Для выхода введи: выход\n")

    while True:
        user_text = input("Ты: ").strip()

        if not user_text:
            continue

        if user_text.lower() in {"выход", "exit", "quit"}:
            print("ZhenyaAI: До встречи!")
            break

        prompt = make_prompt(user_text)

        try:
            answer = generate_answer(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
            )
            print(f"ZhenyaAI: {answer}\n")

        except ValueError:
            print(
                "ZhenyaAI: В вопросе есть символы, которых модель ещё не изучала.\n"
                "Добавь этот текст в core/data/train.txt, переобучи модель "
                "и попробуй снова.\n"
            )


if __name__ == "__main__":
    main()
