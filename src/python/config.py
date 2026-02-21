# coding: utf-8
"""Centralized configuration loader for the GrayCode project.

Reads config.toml from the project root, provides typed dataclasses
for each section, and falls back to hardcoded defaults when the file
or individual fields are absent.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Optional

# Project root: two levels up from src/python/config.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config.toml"


# ── Section dataclasses ──────────────────────────────────────────────


@dataclass(frozen=True)
class PathsConfig:
    pattern_dir: str = "data/graycode_pattern"
    captured_dir: str = "data/captured"


@dataclass(frozen=True)
class CameraConfig:
    av: int = 5
    tv: float = 1 / 15
    iso: int = 100
    image_quality: str = "LJF"
    wait_key_ms: int = 500


@dataclass(frozen=True)
class DecodeConfig:
    black_threshold: int = 50
    white_threshold: int = 5


@dataclass(frozen=True)
class InterpolateC2PConfig:
    inpaint_radius: float = 3.0
    inpaint_method: str = "TELEA"  # "TELEA" or "NS"
    default_method: str = "inpaint"  # "inpaint" or "delaunay"
    csv_precision: int = 4


@dataclass(frozen=True)
class InterpolateP2CConfig:
    csv_precision: int = 4


@dataclass(frozen=True)
class PipelineConfig:
    proj_height: int = 1080
    proj_width: int = 1920
    height_step: int = 1
    width_step: int = 1
    window_pos_x: int = 0
    window_pos_y: int = 0
    default_interpolation_method: str = "delaunay"
    default_input_file: str = "result_c2p.npy"


@dataclass(frozen=True)
class WarpAdvancedConfig:
    eps: float = 1e-8
    large_pos: float = 1e9
    large_neg: float = -1e9
    invalid_coord: float = -1e6


@dataclass(frozen=True)
class WarpConfig:
    default_splat_method: str = "bilinear"
    default_aggregation: str = "mean"
    default_inpaint: str = "none"
    default_inpaint_iter_forward: int = 3
    default_inpaint_iter_backward: int = 5
    default_backward_mode: str = "bilinear"
    default_padding_mode: str = "zeros"
    inpaint_kernel_size: int = 3
    advanced: WarpAdvancedConfig = field(default_factory=WarpAdvancedConfig)


# ── Top-level config container ───────────────────────────────────────


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    interpolate_c2p: InterpolateC2PConfig = field(
        default_factory=InterpolateC2PConfig
    )
    interpolate_p2c: InterpolateP2CConfig = field(
        default_factory=InterpolateP2CConfig
    )
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    warp: WarpConfig = field(default_factory=WarpConfig)


# ── Loading logic ────────────────────────────────────────────────────

_FRACTION_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)\s*$")


def _parse_number(value: object) -> object:
    """文字列の分数表記 (例: "1/15") を float に変換する。

    int や float はそのまま返す。分数表記でない文字列もそのまま返す。
    """
    if isinstance(value, str):
        m = _FRACTION_RE.match(value)
        if m:
            return float(Fraction(m.group(1)) / Fraction(m.group(2)))
    return value


def _build_section(cls: type, data: dict, key: str):
    """Build a dataclass from a TOML sub-dict, ignoring unknown keys."""
    section = data.get(key, {})
    valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: _parse_number(v) for k, v in section.items() if k in valid_fields}
    return cls(**filtered)


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """Load configuration from a TOML file.

    Falls back to compiled-in defaults if the file does not exist
    or if individual fields are absent.
    """
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH

    if not config_path.exists():
        return AppConfig()

    with open(config_path, "rb") as f:
        data = dict(tomllib.load(f))

    # Build warp section specially (has nested 'advanced')
    warp_data = dict(data.get("warp", {}))
    warp_advanced_data = warp_data.pop("advanced", {})

    warp_valid = {
        f.name for f in WarpConfig.__dataclass_fields__.values()
    } - {"advanced"}
    warp_filtered = {k: v for k, v in warp_data.items() if k in warp_valid}

    adv_valid = {f.name for f in WarpAdvancedConfig.__dataclass_fields__.values()}
    adv_filtered = {k: v for k, v in warp_advanced_data.items() if k in adv_valid}

    warp_cfg = WarpConfig(
        advanced=WarpAdvancedConfig(**adv_filtered), **warp_filtered
    )

    return AppConfig(
        paths=_build_section(PathsConfig, data, "paths"),
        camera=_build_section(CameraConfig, data, "camera"),
        decode=_build_section(DecodeConfig, data, "decode"),
        interpolate_c2p=_build_section(InterpolateC2PConfig, data, "interpolate_c2p"),
        interpolate_p2c=_build_section(InterpolateP2CConfig, data, "interpolate_p2c"),
        pipeline=_build_section(PipelineConfig, data, "pipeline"),
        warp=warp_cfg,
    )


# ── Module-level singleton ───────────────────────────────────────────

_config: Optional[AppConfig] = None


def get_config(config_path: Optional[Path] = None) -> AppConfig:
    """Return the cached AppConfig, loading it on first call."""
    global _config
    if _config is None:
        _config = load_config(config_path)
    return _config


def reload_config(config_path: Optional[Path] = None) -> AppConfig:
    """Force-reload configuration (useful for tests)."""
    global _config
    _config = load_config(config_path)
    return _config
