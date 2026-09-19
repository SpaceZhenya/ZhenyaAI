class CharacterTokenizer:
    """
    Простой токенизатор: один символ = один токен.
    Он подходит для первой собственной учебной модели.
    """

    def __init__(self, text: str) -> None:
        characters = sorted(set(text))

        self.characters = characters
        self.char_to_id = {
            character: index
            for index, character in enumerate(characters)
        }
        self.id_to_char = {
            index: character
            for index, character in enumerate(characters)
        }

    @property
    def vocab_size(self) -> int:
        return len(self.characters)

    def encode(self, text: str) -> list[int]:
        unknown = [
            character
            for character in text
            if character not in self.char_to_id
        ]

        if unknown:
            raise ValueError(
                "В тексте есть символы, которых не было в обучающем датасете: "
                f"{''.join(sorted(set(unknown)))}"
            )

        return [
            self.char_to_id[character]
            for character in text
        ]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(
            self.id_to_char[token_id]
            for token_id in token_ids
        )

    def save(self, path: str) -> None:
        import json

        with open(path, "w", encoding="utf-8") as file:
            json.dump(
                {"characters": self.characters},
                file,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "CharacterTokenizer":
        import json

        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        instance = cls("")
        instance.characters = data["characters"]
        instance.char_to_id = {
            character: index
            for index, character in enumerate(instance.characters)
        }
        instance.id_to_char = {
            index: character
            for index, character in enumerate(instance.characters)
        }

        return instance
