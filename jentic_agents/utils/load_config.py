import tomllib
from pathlib import Path
from dacite import from_dict
from .config import Config

def load_config():
    config_path = Path(__file__).parents[2] / "config.toml"
    with open(config_path, "rb") as f:
        config_dict = tomllib.load(f)
    return from_dict(Config, config_dict)