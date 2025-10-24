# data_loader.py
# Marek Hric 

import pandas as pd
import librosa as lb
import numpy as np
from pathlib import Path
import ast
from tqdm import tqdm

class SMDataLoader:
    def __init__(self, path: Path, sr: int, fl: int, fh: int):
        """
        path - absolute path to dataset root, expects reference.csv to be at root
        """
        self._path: Path = path

        self._sr: int = sr
        self._fl: int = fl
        self._fh: int = fh
        self._prefixes: dict[str,list[str]] = {
                "train": ['train/speech/', 'train/m+s/', 'train/music/'],
                "test": ['test/speech/', 'test/music/novocals/', 'test/music/vocals/']
        }
        print("SMDataLoader created")


    def load(self, train: bool) -> list[tuple[list[int], list[np.ndarray]]]:
        """
        returns ref and frames
        """
        def _load_and_resample(file_path: Path) -> np.ndarray:
            return lb.load(file_path, sr=self._sr)[0] # mono

        df = pd.read_csv(self._path / Path('reference.csv'))
        df = df[df['file'].str.startswith(
            tuple(self._prefixes['train' if train else 'test'])
        )]

        result = []
        for _, row in tqdm(df.iterrows(), desc=f"Loading files [{'train' if train else 'test'}]"):
            audio = _load_and_resample(self._path / Path(row['file']))
            result.append((ast.literal_eval(row['reference']), self._framing(audio)))

        return result

    def _framing(self, s: np.ndarray) -> list[np.ndarray]:
        fl = self._fl
        fh = self._fh
        Nf = int(1 + np.floor((len(s) - fl) / fh))  # number of frames

        hann_win = np.hanning(fl)

        return [s[i * fh:i * fh + fl] * hann_win for i in range(Nf)]

