import numpy as np
import config


class FeatExtractor:
    def __init__(self):
        self.cfg = config.get_config().fext
        pass

    def extract(self, frame: np.ndarray) -> np.ndarray:
        return np.array([])
