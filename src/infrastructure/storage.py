import json
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from domain.types import JsonType


class JsonStorage:
    def __init__(self, path: str | Path, format: bool = False):
        self.path = Path(path)
        self.format = format

    def save(self, obj: JsonType | BaseModel | pd.DataFrame) -> None:
        if isinstance(obj, BaseModel):
            obj = obj.model_dump(mode="json")

        if isinstance(obj, pd.DataFrame):
            obj = obj.reset_index().to_dict(orient="records")

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.path.write_text(
            json.dumps(
                obj,
                default=str,
                indent=2 if self.format else None,
            ),
            encoding="utf-8",
        )

    def load(self) -> JsonType:
        if not self.path.exists():
            return {}

        return json.loads(
            self.path.read_text(
                encoding="utf-8",
            )
        )

    def remove(self) -> None:
        if self.path.exists():
            self.path.unlink()
