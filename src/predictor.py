"""
Various code predictor models.
"""

from src.backend import LLMBackend


class BasePredictor:
    def _predict(self, code: str) -> str:
        raise NotImplementedError

    def __call__(self, codes: list[str]) -> list[str]:
        raise NotImplementedError


class LLMPredictor(BasePredictor):
    def __init__(self, model_name: str, *args, **kwargs) -> None:
        self.llm_model = LLMBackend(model_name)

    def _predict(self, code: str) -> str:
        with open("./prompts/description_generation.txt") as f:
            prompt = f.read()

        messages = [
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": f"#Code\n{code}",
            },
        ]

        return self.llm_model.query(messages)

    def __call__(self, codes: list[str]) -> list[str]:
        return [self._predict(code) for code in codes]
