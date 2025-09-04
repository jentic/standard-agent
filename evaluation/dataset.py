from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple


def load_dataset(path: str | Path) -> Iterator[Tuple[str, str, Optional[object], Dict]]:
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                item_id = str(obj.get("id", idx))
                goal = obj.get("goal", "")
                expected = obj.get("expected")
                metadata = obj.get("metadata", {})
                yield item_id, goal, expected, metadata
    elif p.suffix.lower() == ".csv":
        with p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                item_id = row.get("id") or str(idx)
                goal = row.get("goal", "")
                expected = row.get("expected")
                yield item_id, goal, expected, {}
    else:
        raise ValueError(f"Unsupported dataset format: {p.suffix}")


__all__ = ["load_dataset"]


