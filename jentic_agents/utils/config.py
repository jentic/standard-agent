import tomllib
from typing import Any, Dict
from pathlib import Path

_CONFIG_CACHE: Dict[str, Any] = {}
CONFIG_FILE = Path(__file__).parents[2] / "config.toml"

def load_config() -> Dict[str, Any]:
    global _CONFIG_CACHE
    if not _CONFIG_CACHE:
        config_path = Path(CONFIG_FILE).resolve()
        try:
            with open(config_path, "rb") as f:
                full_config = tomllib.load(f)
            config = full_config

            if "llm" in config and "provider" in config["llm"] and "model" in config["llm"]:
                provider = config["llm"]["provider"]
                model = config["llm"]["model"]
                if not model.startswith(provider + "/"):
                    config["llm"]["model"] = f"{provider}/{model}"

            _CONFIG_CACHE = config
        except Exception as e:
            raise RuntimeError(f"Failed to load config from config.toml: {e}")
    return _CONFIG_CACHE

def get_config_value(*keys, default=None) -> Any:
    """Get a nested config value by keys, e.g. get_config_value('llm', 'model')."""
    config = load_config()
    for key in keys:
        if isinstance(config, dict) and key in config:
            config = config[key]
        else:
            return default
    return config