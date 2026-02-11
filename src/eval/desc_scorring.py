import re
from typing import override

import numpy as np
from pydantic import BaseModel, Field

from src.predictor.base import BasePredictor
from src.eval.base import BaseEvaluator
from src.backend import LLMBackend


class DescriptionEvaluator(BaseEvaluator):
    """LLM as a judge for code description scorring"""

    class Metrics(BaseModel):
        intention: int = Field(ge=1, le=4)
        technical_specs: int = Field(ge=1, le=4)
        abstraction_level: int = Field(ge=1, le=4)
        implementation_ease: int = Field(ge=1, le=4)

    def __init__(
        self,
        model: BasePredictor,
        judges: list[str],
        dataloader,
        save_preds: bool = False,
        experiment_name: str | None = None,
    ) -> None:
        super().__init__(model, dataloader)
        self.model: BasePredictor = model
        self.judges: list[LLMBackend] = [LLMBackend(llm) for llm in judges]
        self.results: dict[str, dict[str, list[int]]] = {
            judge: {
                "intention": [],
                "technical_specs": [],
                "abstraction_level": [],
                "implementation_ease": [],
            }
            for judge in judges
        }
        self.batch_count: int = 0
        self.save_preds: bool = save_preds
        if save_preds and experiment_name is None:
            raise ValueError(
                "`experiment_name` can't be none when `save_preds` is set to True"
            )
        self.experiment_name: str | None = experiment_name
        self.computed_metrics: dict[str, dict[str, float]] | None = None

    def _parse_response(self, res: str) -> Metrics:
        regex = re.compile(r"(?:Output: )({.*})")
        m = re.search(regex, res)
        return self.Metrics.model_validate_json(m.group(1) if m else "")  # type: ignore

    @override
    def reset(self) -> None:
        for judge in self.results:
            for metric in self.results[judge]:
                self.results[judge][metric] = []
        self.batch_count = 0
        self.computed_metrics = None

    @override
    def process_batch(self, batch) -> None:
        with open("./prompts/description_scorring.txt") as f:
            prompt = f.read()

        x, y = batch["code"], batch["desc"]
        preds = self.model(x)

        for fake, true in zip(preds, y):
            message = [
                {
                    "role": "user",
                    "content": prompt.format(desc=fake),
                }
            ]

            raw_response = None
            metrics = None
            try:
                for judge in self.judges:
                    raw_response = judge.query(message)
                    metrics = self._parse_response(raw_response)
                    self.results[judge.alias]["intention"].append(metrics.intention)
                    self.results[judge.alias]["technical_specs"].append(
                        metrics.technical_specs
                    )
                    self.results[judge.alias]["abstraction_level"].append(
                        metrics.abstraction_level
                    )
                    self.results[judge.alias]["implementation_ease"].append(
                        metrics.implementation_ease
                    )

            except ValueError as e:
                print(e)

        self.batch_count += 1

    @override
    def compute_metrics(self) -> dict[str, dict[str, float]]:
        if self.computed_metrics:
            return self.computed_metrics

        results = {}
        for judge in self.results:
            results[judge] = {}
            for metric in self.results[judge]:
                results[judge][metric] = float(np.mean(self.results[judge][metric]))

        self.computed_metrics = results
        return results
