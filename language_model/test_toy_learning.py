from __future__ import annotations

import torch
import torch.nn as nn


SEED = 42
VOCAB_SIZE = 8
EMBEDDING_DIM = 16
STEPS = 300
LEARNING_RATE = 0.03


class TinyNextTokenModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
        )

        self.head = nn.Linear(
            in_features=EMBEDDING_DIM,
            out_features=VOCAB_SIZE,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        return self.head(embeddings)


def main() -> None:
    torch.manual_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("ТЕСТ ОБУЧЕНИЯ НА ПРОСТОЙ ЗАКОНОМЕРНОСТИ")
    print("-" * 55)
    print(f"Устройство: {device}")
    print(
        "Правило: следующий токен = "
        "(текущий токен + 1) % 8\n"
    )

    input_ids = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7],
        dtype=torch.long,
        device=device,
    )

    targets = torch.tensor(
        [1, 2, 3, 4, 5, 6, 7, 0],
        dtype=torch.long,
        device=device,
    )

    model = TinyNextTokenModel().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )

    loss_function = nn.CrossEntropyLoss()

    model.train()

    initial_loss = 0.0
    final_loss = 0.0

    for step in range(1, STEPS + 1):
        optimizer.zero_grad(set_to_none=True)

        logits = model(input_ids)

        loss = loss_function(
            logits,
            targets,
        )

        loss.backward()
        optimizer.step()

        if step == 1:
            initial_loss = loss.item()

        if step in {1, 10, 50, 100, 200, 300}:
            print(
                f"Шаг {step:>3}: "
                f"loss = {loss.item():.6f}"
            )

        final_loss = loss.item()

    model.eval()

    with torch.no_grad():
        logits = model(input_ids)
        predictions = torch.argmax(
            logits,
            dim=-1,
        )

    correct = int(
        (predictions == targets).sum().item()
    )

    accuracy = correct / len(targets) * 100

    print("\nРЕЗУЛЬТАТ")
    print("-" * 55)
    print(f"Начальный loss: {initial_loss:.6f}")
    print(f"Финальный loss: {final_loss:.6f}")
    print(f"Правильных ответов: {correct}/{len(targets)}")
    print(f"Точность: {accuracy:.1f}%")

    print("\nПРОГНОЗЫ")
    print("-" * 55)

    for input_id, target, prediction in zip(
        input_ids.tolist(),
        targets.tolist(),
        predictions.tolist(),
    ):
        status = "OK" if target == prediction else "ОШИБКА"

        print(
            f"{input_id} -> ожидалось {target}, "
            f"получено {prediction}: {status}"
        )

    if final_loss >= initial_loss:
        raise RuntimeError(
            "Loss не снизился: проверь цикл обучения."
        )

    if correct != len(targets):
        raise RuntimeError(
            "Модель не выучила простую закономерность."
        )

    print("\nТЕСТ ПРОЙДЕН")
    print(
        "Модель способна учиться: "
        "loss снизился, а правило выучено полностью."
    )


if __name__ == "__main__":
    main()
