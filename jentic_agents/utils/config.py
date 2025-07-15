import tomllib
from typing import Any, Dict
from pathlib import Path

_CONFIG_CACHE: Dict[str, Any] = {}
CONFIG_FILE = Path(__file__).parents[2] / "config.toml"

def _load_config() -> Dict[str, Any]:
    global _CONFIG_CACHE
    if not _CONFIG_CACHE:
        config_path = Path(CONFIG_FILE).resolve()
        try:
            with open(config_path, "rb") as f:
                full_config = tomllib.load(f)
            config = full_config.get("tool", {}).get("actbots", {})

            if "llm" in config and "provider" in config["llm"] and "model" in config["llm"]:
                provider = config["llm"]["provider"]
                model = config["llm"]["model"]
                if not model.startswith(provider + "/"):
                    config["llm"]["model"] = f"{provider}/{model}"

            if (
                "memory" in config
                and "llm_provider" in config["memory"]
                and "llm_model" in config["memory"]
            ):
                provider = config["memory"]["llm_provider"]
                model = config["memory"]["llm_model"]
                if not model.startswith(provider + "/"):
                    config["memory"]["llm_model"] = f"{provider}/{model}"

            _CONFIG_CACHE = config
        except Exception as e:
            raise RuntimeError(f"Failed to load config from config.toml: {e}")
    return _CONFIG_CACHE


def get_config() -> Dict[str, Any]:
    """Return the full config as a dict."""
    return _load_config()

def get_config_value(*keys, default=None) -> Any:
    """Get a nested config value by keys, e.g. get_config_value('llm', 'model')."""
    config = _load_config()
    for key in keys:
        if isinstance(config, dict) and key in config:
            config = config[key]
        else:
            return default
    return config

def get_logging_config() -> Dict[str, Any]:
    """Get the logging configuration (backward compatibility)."""
    try:
        config_path = Path(CONFIG_FILE).resolve()
        with open(config_path, "rb") as f:
            full_config = tomllib.load(f)
        return full_config.get("tool", {}).get("actbots", {}).get("logging", {})
    except Exception:
        return {} 