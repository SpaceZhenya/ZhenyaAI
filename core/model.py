import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZhenyaTransformer(nn.Module):
    """
    Небольшая GPT-подобная языковая модель.
    Она предсказывает следующий токен текста.
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int = 128,
        embedding_size: int = 128,
        heads: int = 4,
        layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(block_size, embedding_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=heads,
            dim_feedforward=embedding_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=layers,
        )

        self.final_norm = nn.LayerNorm(embedding_size)
        self.output = nn.Linear(embedding_size, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, sequence_length = tokens.shape

        if sequence_length > self.block_size:
            raise ValueError(
                f"Текст слишком длинный: максимум {self.block_size} токенов."
            )

        positions = torch.arange(
            sequence_length,
            device=tokens.device,
        )

        token_vectors = self.token_embedding(tokens)
        position_vectors = self.position_embedding(positions)

        x = token_vectors + position_vectors

        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=tokens.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        x = self.transformer(x, mask=causal_mask)
        x = self.final_norm(x)

        logits = self.output(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        new_tokens: int = 100,
        temperature: float = 0.8,
    ) -> torch.Tensor:
        self.eval()

        for _ in range(new_tokens):
            context = tokens[:, -self.block_size:]
            logits, _ = self(context)

            next_logits = logits[:, -1, :] / max(temperature, 0.01)
            probabilities = F.softmax(next_logits, dim=-1)

            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )

            tokens = torch.cat((tokens, next_token), dim=1)

        return tokens

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def create_small_model(vocab_size: int) -> ZhenyaTransformer:
    """
    Стартовая маленькая конфигурация модели для обычного компьютера.
    """
    return ZhenyaTransformer(
        vocab_size=vocab_size,
        block_size=128,
        embedding_size=128,
        heads=4,
        layers=4,
        dropout=0.1,
    )


if __name__ == "__main__":
    test_model = create_small_model(vocab_size=256)

    example_tokens = torch.randint(
        low=0,
        high=256,
        size=(2, 16),
    )

    logits, loss = test_model(
        example_tokens,
        example_tokens,
    )

    print(f"Параметров в модели: {test_model.parameter_count():,}")
    print(f"Форма logits: {logits.shape}")
    print(f"Тестовая loss: {loss.item():.4f}")
