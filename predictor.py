"""
Various code predictor models.
"""

from backend import LLMBackend


class BasePredictor:
    def _predict(self, code: str) -> str:
        raise NotImplementedError

    def __call__(self, codes: str) -> list[str]:
        raise NotImplementedError


class LLMPredictor(BasePredictor):
    def __init__(self, model_name: str, *args, **kwargs) -> None:
        self.llm_model = LLMBackend(model_name)

    def _predict(self, code: str) -> str:
        with open("./prompts/predictor.txt") as f:
            prompt = f.read()

        messages = [
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": code,
            },
        ]

        return self.llm_model.query(messages)

    def __call__(self, codes: str) -> list[str]:
        return [self._predict(code) for code in codes]
