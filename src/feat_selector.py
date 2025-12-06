import numpy as np
import config


class FeatSelector:
    def __init__(self):
        self.cfg = config.get_config().fsel
        pass

    def extract(self, frame: np.ndarray) -> np.ndarray:
        return np.array([])
