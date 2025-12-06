# config.py
# Marek Hric

from __future__ import annotations
from dataclasses import dataclass, fields, is_dataclass
from typing import Optional, Any, Type, TypeVar, Dict
import sys
import yaml

T = TypeVar("T")


@dataclass
class Defaults:
    sample_rate: int
    frame_length: int
    hop_length: int
    max_buffer: int
    channels: int
    n_fft: int


@dataclass
class FeatureConfig:
    enable: bool


@dataclass
class MFCCConfig(FeatureConfig):
    n_mfcc: int = 13


@dataclass
class BandConfig:
    lbound: int
    ubound: int


@dataclass
class BandEnergyRatioConfig(FeatureConfig):
    lband: BandConfig
    uband: BandConfig


@dataclass
class SpecRolloffPointConfig(FeatureConfig):
    thr: float = 0.85


@dataclass
class FeatExtractorConfig:
    verbose: bool
    mfcc: MFCCConfig
    mfcc_diff_norm: MFCCConfig
    st_energy: FeatureConfig
    zcr: FeatureConfig
    band_energy_ratio: BandEnergyRatioConfig
    s_rolloff_point: SpecRolloffPointConfig
    s_centroid: FeatureConfig
    s_spread: FeatureConfig
    s_flux: FeatureConfig


@dataclass
class FeatSelectorConfig:
    verbose: bool


@dataclass
class ModelConfig:
    verbose: bool
    name: str
    aggregate: bool


def from_dict(cls: Type[T], data: Any) -> T:
    """
    Recursively load a nested dict into a dataclass.
    """
    if not is_dataclass(cls):
        return data  # base case

    field_types = {f.name: f.type for f in fields(cls)}
    init_values = {}
    for name, typ in field_types.items():
        if name in data:
            value = data[name]
            init_values[name] = from_dict(typ, value)  # type: ignore error[invalid-argument-type]
    return cls(**init_values)


class Config:
    """
    mode: str (train, eval, run)
    model: ModelConfig
    fext: FeatExtractorConfig
    fsel: Optional[FeatSelectorConfig]
    defaults: Defaults
    """

    def __init__(self, args: Dict[str, str | bool]):
        self.mode: str = args["mode"]
        model_name = args["model"]
        assert isinstance(model_name, str)
        conf_file = args.get("config_file", "config.yaml")
        assert isinstance(conf_file, str)
        verbose = args["verbose"]
        assert isinstance(verbose, bool)

        try:
            with open(conf_file) as f:
                conf_dict = yaml.safe_load(f)
        except FileNotFoundError as e:
            print(f"Config file not found: {conf_file}", file=sys.stderr)
            raise e

        sr = conf_dict["defaults"]["sample_rate"]
        self.defaults = Defaults(
            sample_rate=sr,
            frame_length=conf_dict["defaults"]["frame_length_ms"] * int(sr / 1000),
            hop_length=conf_dict["defaults"]["hop_length_ms"] * int(sr / 1000),
            max_buffer=conf_dict["defaults"]["max_buffer_ms"] * int(sr / 1000),
            channels=conf_dict["defaults"]["channels"],
            n_fft=conf_dict["defaults"]["n_fft"],
        )

        feat_dict = conf_dict["models"][model_name].get("features", {})
        feat_dict = {"verbose": verbose, **feat_dict}
        self.fext: FeatExtractorConfig = from_dict(FeatExtractorConfig, feat_dict)

        selector_cfg = conf_dict["models"][model_name].get("selector", {})
        self.fsel: Optional[FeatSelectorConfig] = None
        if selector_cfg.get("enable", False):
            self.fsel = FeatSelectorConfig(verbose=verbose)

        self.model: ModelConfig = ModelConfig(
            verbose=verbose,
            name=model_name,
            aggregate=conf_dict["models"][model_name]["aggregate"],
        )

        assert isinstance(self.mode, str)
        assert isinstance(self.model, ModelConfig)
        assert isinstance(self.fext, FeatExtractorConfig)
        assert isinstance(self.fsel, Optional[FeatSelectorConfig])
        assert isinstance(self.defaults, Defaults)


_cfg: Optional[Config] = None


def init_config(args: Dict[str, str | bool]):
    global _cfg
    if _cfg is not None:
        return
    _cfg = Config(args)


def get_config() -> Config:
    if _cfg is None:
        raise RuntimeError("Config accessed before initialization")

    return _cfg
