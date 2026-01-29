from transformers.data import metrics
from backend import LLMBackend
from model_mapping import model_name_mapping

from abc import ABC, abstractmethod
from typing import Any, Optional

from bert_score import BERTScorer
from numpy import average
from pydantic import BaseModel


type EvalMetrics = dict


class BaseEvaluator(ABC):
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def process_batch(self, batch) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute_metrics(self) -> dict:
        raise NotImplementedError

    def evaluate(self):
        self.reset()
        self.model.eval()

        for batch in self.dataloader:
            self.process_batch(batch)

        return self.compute_metrics()


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


class LLMJudgeEvaluator(BaseEvaluator):
    class Metrics(BaseModel):
        # TODO: Implement this
        pass

    def __init__(self, model, dataloader, judge_model) -> None:
        super().__init__(model, dataloader)
        self.judge = LLMBackend(judge_model)

    def _parse_response(self, res: str) -> Metrics:
        # TODO: Implement
        return self.Metrics()

    def reset(self) -> None:
        # TODO: Implement
        pass

    def process_batch(self, batch) -> None:
        with open("./prompts/judge.txt") as f:
            prompt = f.read()

        metrics: list[LLMJudgeEvaluator.Metrics] = []
        x, y = batch

        preds = self.model(x)

        for cand, ref in zip(preds, y):
            message = [
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": f"True description:\n{ref}",
                },
                {
                    "role": "user",
                    "content": f"Generated description:\n{cand}",
                },
            ]

            metrics.append(self._parse_response(self.judge.query(message)))

    def compute_metrics(self) -> dict:
        pass  # TODO: implement
