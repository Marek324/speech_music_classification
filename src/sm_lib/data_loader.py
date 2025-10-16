# data_loader.py
# Marek Hric 

import pandas as pd
import librosa as lb
import numpy as np
from pathlib import Path
from lib.defaults import SAMPLE_RATE
import ast

class SMDataLoader:
    def __init__(self, path: Path, train: bool = True):
        """
        path - absolute path to reference.csv (use pathlib.Path.absolute())
        """
        self._path: Path = path
        self._speech_prefixes: list[str] = ['train/speech/', 'train/m+s/'] if train else ['test/speech/']
        self._music_prefixes: list[str] = ['train/music/'] if train else ['test/music/novocals/', 'test/music/vocals/']

        self.data: list[tuple[int, np.ndarray]] = self._load()


    def _load(self) -> list[tuple[int, np.ndarray]]:
        def _load_and_resample(file_path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
            return lb.load(file_path, sr=target_sr)[0] # mono

        def _target_class(ref: list[int]) -> int:
            """
            only 1 value as silence is determined elsewhere and there are no mixed-class files in dataset
            """
            return 1 if any(r == 1 for r in ref) else 2

        def _load_class(df: pd.DataFrame, prefixes: list[str]) :
            fdf: pd.DataFrame = df[df['file'].str.startswith(tuple(prefixes))]

            result: list[tuple[int, np.ndarray]] = []
            for _, row in fdf.iterrows():
                target: int = _target_class(ast.literal_eval(row['reference']))
                audio: np.ndarray = _load_and_resample(self._path / Path(row['file']))
                result.append((target, audio))

            return result

        df: pd.DataFrame = pd.read_csv(self._path)
        return _load_class(df, self._speech_prefixes) + _load_class(df, self._music_prefixes)

