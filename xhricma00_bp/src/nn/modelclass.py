# nn/modelclass.py
# Author: Marek Hric
# The help of code assistant was used during implementation of this file.

from abc import ABC, abstractmethod

from ..evaluator import EvalResults


class NNModelClass(ABC):
    """Base class for neural network speech/music classifiers."""

    name: str

    @abstractmethod
    def train(self, **kwargs) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def evaluate(self, save_to_file: bool = True) -> EvalResults:
        """Evaluate the model on the test set."""
        pass

    @abstractmethod
    def smoke_test(self) -> None:
        """Quick sanity check (forward pass / minimal run)."""
        pass
