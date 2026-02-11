"""
Various code predictor models.
"""

import os
import json
from abc import ABC, abstractmethod


class BasePredictor(ABC):
    def __init__(self, experiment_name: str) -> None:
        self.experiment_name: str = experiment_name
        self.batch_number: int = 0

    @abstractmethod
    def eval(self) -> None:
        raise NotImplementedError

    def _save_preds(self, preds, x) -> None:
        os.makedirs("./outputs", exist_ok=True)
        os.makedirs(f"./outputs/{self.experiment_name}", exist_ok=True)
        with open(
            f"./outputs/{self.experiment_name}/{self.batch_number}.json",
            "w",
        ) as f:
            json.dump(
                {
                    "x": x,
                    "preds": preds,
                },
                f,
            )
            self.batch_number += 1

    @abstractmethod
    def _predict(self, code: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, codes: list[str]) -> list[str]:
        raise NotImplementedError
