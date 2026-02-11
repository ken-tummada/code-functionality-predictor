from src.backend import LLMBackend
from src.predictor.base import BasePredictor


class DescriptionLLMPredictor(BasePredictor):
    def __init__(self, model: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.llm_model = LLMBackend(model)

    def _predict(self, code: str) -> str:
        with open("./prompts/description_generation.txt") as f:
            prompt = f.read()

        messages = [
            {
                "role": "user",
                "content": prompt.format(code=code),
            },
        ]

        return self.llm_model.query(messages)

    def eval(self) -> None:
        pass

    def __call__(self, codes: list[str]) -> list[str]:
        preds = [self._predict(code) for code in codes]
        self._save_preds(preds, codes)
        return preds
