import tosholi
from pathlib import Path
from .config import Config

CONFIG_FILE = Path(__file__).parents[2] / "config.toml"

def load_config() -> Config:
    with open(CONFIG_FILE, "rb") as f:
        config = tosholi.load(Config, f)

    return config
