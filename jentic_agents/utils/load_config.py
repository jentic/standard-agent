import tosholi
from pathlib import Path
from .config import Config

CONFIG_FILE = Path(__file__).parents[2] / "config.toml"

def load_config() -> Config:
    with open(CONFIG_FILE, "rb") as f:
        config = tosholi.load(Config, f)

    if config.llm.provider and config.llm.model and not config.llm.model.startswith(config.llm.provider + "/"):
        config.llm.model = f"{config.llm.provider}/{config.llm.model}"

    return config
