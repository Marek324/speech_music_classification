# evaluator.py
# Marek Hric

from frame_dataset import FrameDataset
from modelclass import ModelClass


class Evaluator:
    def __init__(self, fd: FrameDataset):
        self.ds = fd

    def eval(self, model: ModelClass):
        pass
