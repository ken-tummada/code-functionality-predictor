from abc import ABC, abstractmethod

from tqdm import tqdm


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

        for batch in tqdm(self.dataloader, desc="Generating predictions"):
            self.process_batch(batch)

        return self.compute_metrics()
