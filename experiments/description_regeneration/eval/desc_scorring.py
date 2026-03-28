import json
import re
import os
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from src.predictor.base import BasePredictor
from src.eval.base import BaseEvaluator
from src.backend import LLMBackend


class DescriptionScorrer(BaseEvaluator):
    """LLM as a judge for code description scorring"""

    class Metrics(BaseModel):
        intention: int = Field(ge=1, le=4)
        technical_specs: int = Field(ge=1, le=4)
        abstraction_level: int = Field(ge=1, le=4)
        implementation_ease: int = Field(ge=1, le=4)

    def __init__(
        self,
        model: BasePredictor,
        dataloader,
        judge: LLMBackend,
        save_preds=False,
        experiment_name: str | None = None,
    ) -> None:
        super().__init__(model, dataloader)
        self.judge = judge
        self.results = {
            "intention": [],
            "technical_specs": [],
            "abstraction_level": [],
            "implementation_ease": [],
        }
        self.batch_count = 0
        self.save_preds = save_preds
        if save_preds and experiment_name is None:
            raise ValueError(
                "`experiment_name` can't be none when `save_preds` is set to True"
            )
        self.experiment_name = experiment_name
        self.computed_metrics = None

    def _parse_response(self, res: str) -> Metrics:
        regex = re.compile(r"(?:Output: )({.*})")
        m = re.search(regex, res)
        return self.Metrics.model_validate_json(m.group(1) if m else "")  # type: ignore

    def reset(self) -> None:
        for key in self.results:
            self.results[key] = []
        self.batch_count = 0
        self.computed_metrics = None

    def process_batch(self, batch) -> None:
        with open("./prompts/description_scorring.txt") as f:
            prompt = f.read()

        x, y = batch["code"], batch["desc"]
        preds = self.model(x)

        if self.save_preds:
            os.makedirs("./outputs", exist_ok=True)
            os.makedirs(f"./outputs/{self.experiment_name}", exist_ok=True)
            with open(
                f"./outputs/{self.experiment_name}/{self.batch_count}.json",
                "w",
            ) as f:
                json.dump(
                    {
                        "x": x,
                        "preds": preds,
                    },
                    f,
                )

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
                raw_response = self.judge.query(message)
                metrics = self._parse_response(raw_response)
                self.results["intention"].append(metrics.intention)
                self.results["technical_specs"].append(metrics.technical_specs)
                self.results["abstraction_level"].append(metrics.abstraction_level)
                self.results["implementation_ease"].append(metrics.implementation_ease)

            except ValueError as e:
                print(e)

        self.batch_count += 1

    def compute_metrics(self) -> dict:
        if self.computed_metrics:
            return self.computed_metrics

        results = {}
        for key in self.results:
            results[key] = np.mean(self.results[key])

        self.computed_metrics = results
        return results
