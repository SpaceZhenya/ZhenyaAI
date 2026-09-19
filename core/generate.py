from pathlib import Path

import torch

from core.model import create_small_model
from core.tokenizer import CharacterTokenizer


ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
MODEL_PATH = CHECKPOINT_DIR / "zhenyaai.pt"
TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer.json"

PROMPT = "ZhenyaAI"
NEW_TOKENS = 300
TEMPERATURE = 0.7


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Веса не найдены. Сначала запусти обучение:\n"
            "python -m core.train"
        )

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            "Токенизатор не найден. Сначала запусти обучение:\n"
            "python -m core.train"
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

    try:
        prompt_ids = tokenizer.encode(PROMPT)
    except ValueError as error:
        print(error)
        print("\nИспользуй в PROMPT только символы из core/data/train.txt.")
        return

    input_tokens = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    generated_tokens = model.generate(
        tokens=input_tokens,
        new_tokens=NEW_TOKENS,
        temperature=TEMPERATURE,
    )

    result_ids = generated_tokens[0].tolist()
    result_text = tokenizer.decode(result_ids)

    print("\n--- Ответ ZhenyaAI ---\n")
    print(result_text)
    print("\n----------------------\n")


if __name__ == "__main__":
    main()
