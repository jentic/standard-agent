from pathlib import Path
import yaml


def load_prompts(profile: str, required_keys: list[str]) -> dict[str, str]:
    path = Path(__file__).parent / f"{profile}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    missing = [k for k in required_keys if k not in data or not isinstance(data[k], str) or not data[k].strip()]
    if missing:
        raise KeyError(f"Missing/empty prompt keys in {path}: {missing}")
    return data


