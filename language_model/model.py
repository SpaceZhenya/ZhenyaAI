from __future__ import annotations

import math

import torch
from torch import nn

from language_model.config import ModelConfig


class CausalSelfAttention(nn.Module):
    """
    Self-attention, который запрещает смотреть в будущие токены.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        if config.embedding_dim % config.num_heads != 0:
            raise ValueError(
                "embedding_dim должен делиться на num_heads."
            )

        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads

        self.query = nn.Linear(
            config.embedding_dim,
            config.embedding_dim,
        )
        self.key = nn.Linear(
            config.embedding_dim,
            config.embedding_dim,
        )
        self.value = nn.Linear(
            config.embedding_dim,
            config.embedding_dim,
        )

        self.output = nn.Linear(
            config.embedding_dim,
            config.embedding_dim,
        )

        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)

        causal_mask = torch.tril(
            torch.ones(
                config.context_length,
                config.context_length,
                dtype=torch.bool,
            )
        )

        self.register_buffer(
            "causal_mask",
            causal_mask.view(
                1,
                1,
                config.context_length,
                config.context_length,
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embedding_dim = x.shape

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        query = query.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        key = key.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        value = value.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        attention_scores = query @ key.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        mask = self.causal_mask[
            :,
            :,
            :sequence_length,
            :sequence_length,
        ]

        attention_scores = attention_scores.masked_fill(
            ~mask,
            torch.finfo(attention_scores.dtype).min,
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )

        attention_weights = self.attention_dropout(
            attention_weights,
        )

        attended = attention_weights @ value

        attended = attended.transpose(1, 2).contiguous()

        attended = attended.view(
            batch_size,
            sequence_length,
            embedding_dim,
        )

        output = self.output(attended)
        output = self.output_dropout(output)

        return output


class FeedForward(nn.Module):
    """
    Небольшая MLP-сеть внутри Transformer-блока.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        hidden_dim = config.embedding_dim * 4

        self.network = nn.Sequential(
            nn.Linear(config.embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TransformerBlock(nn.Module):
    """
    Один decoder-style Transformer-блок.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.first_norm = nn.LayerNorm(config.embedding_dim)
        self.attention = CausalSelfAttention(config)

        self.second_norm = nn.LayerNorm(config.embedding_dim)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.first_norm(x))
        x = x + self.feed_forward(self.second_norm(x))

        return x


class ZhenyaLanguageModel(nn.Module):
    """
    Собственная GPT-подобная языковая модель ZhenyaAI.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        config.validate()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
        )

        self.position_embedding = nn.Embedding(
            config.context_length,
            config.embedding_dim,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(
            config.embedding_dim,
        )

        self.language_head = nn.Linear(
            config.embedding_dim,
            config.vocab_size,
            bias=False,
        )

        self.apply(self._initialize_weights)

        self.language_head.weight = self.token_embedding.weight

    def _initialize_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.config.context_length:
            raise ValueError(
                "Длина последовательности превышает context_length: "
                f"{self.config.context_length}."
            )

        positions = torch.arange(
            sequence_length,
            device=input_ids.device,
        )

        token_vectors = self.token_embedding(input_ids)
        position_vectors = self.position_embedding(positions)

        x = token_vectors + position_vectors

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.language_head(x)

        loss = None

        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int | None = 50,
    ) -> torch.Tensor:
        self.eval()

        if temperature <= 0:
            raise ValueError("temperature должен быть больше нуля.")

        for _ in range(max_new_tokens):
            context = input_ids[
                :,
                -self.config.context_length:,
            ]

            logits, _ = self(context)
            next_logits = logits[:, -1, :]
            next_logits = next_logits / temperature

            if top_k is not None:
                top_k = min(top_k, next_logits.size(-1))

                values, _ = torch.topk(
                    next_logits,
                    top_k,
                )

                minimum_value = values[:, [-1]]

                next_logits = torch.where(
                    next_logits < minimum_value,
                    torch.full_like(
                        next_logits,
                        torch.finfo(next_logits.dtype).min,
                    ),
                    next_logits,
                )

            probabilities = torch.softmax(
                next_logits,
                dim=-1,
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )

            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1,
            )

        return input_ids

    def parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
        )


def create_model_from_tokenizer(
    vocab_size: int,
) -> ZhenyaLanguageModel:
    """
    Создаёт модель с текущей конфигурацией проекта.
    """

    config = ModelConfig(
        vocab_size=vocab_size,
    )

    return ZhenyaLanguageModel(config)


def main() -> None:
    config = ModelConfig(
        vocab_size=8_000,
    )

    model = ZhenyaLanguageModel(config)

    batch_size = 2
    sequence_length = 32

    example_input = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, sequence_length),
    )

    example_targets = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, sequence_length),
    )

    logits, loss = model(
        input_ids=example_input,
        targets=example_targets,
    )

    print("Модель успешно создана.")
    print(f"Размер logits: {tuple(logits.shape)}")
    print(f"Количество параметров: {model.parameter_count():,}")
    print(f"Тестовая loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
