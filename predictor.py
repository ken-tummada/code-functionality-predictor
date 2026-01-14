from openai import OpenAI


class BasePredictor:
    def __init__(self) -> None:
        pass

    def predict(self, code: str) -> str:
        return code


class LLMPredictor(BasePredictor):
    def __init__(self, model: str) -> None:
        self.client = OpenAI()
        self.model = model

    def predict(self, code: str) -> str:
        prompt = ""  # TODO: write the prompt
        response = self.client.responses.create(
            model=self.model,
            input=code,  # TODO: format
        )

        return response.output_text
