"""
Various code predictor models.
"""

from abc import ABC, abstractmethod


class BasePredictor(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def eval(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _predict(self, code: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, codes: list[str]) -> list[str]:
        raise NotImplementedError
