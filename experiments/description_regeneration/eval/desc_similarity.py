from abc import ABC, abstractmethod
from typing import Any, Optional

from bert_score import BERTScorer
from numpy import average
from pydantic import BaseModel

from src.eval.base import BaseEvaluator


class BERTScoreEvaluator(BaseEvaluator):
    class Metrics(BaseModel):
        F1: float

    def __init__(self, model, dataloader):
        super().__init__(model, dataloader)
        self.scorer = BERTScorer(lang="en")
        self.f1s = []

    def reset(self):
        self.f1s = []

    def process_batch(self, batch):
        x, y = batch
        preds = self.model(x)
        _, _, F1 = self.scorer.score(preds, y)
        self.f1s.append(F1.mean())

    def compute_metrics(self) -> dict:
        return self.Metrics(F1=average(self.f1s)).model_dump()
