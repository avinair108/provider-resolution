"""Disk cache keyed on the full request, so reruns are free and give the same answers.

A key covers provider, model, prompts, parameters and the sample index. Change any of them and it is a new request.
Entries live under .cache/llm/ (git-ignored) and hold the request alongside the response for auditing.
"""

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from llm.types import Request, Response


def request_key(request: Request, provider: str) -> str:
    canonical = json.dumps(request.key_payload(provider), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


class Cache:
    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> Response | None:
        path = self._path(key)
        if not path.exists():
            return None
        entry = json.loads(path.read_text())
        return Response(**{**entry["response"], "cached": True})

    def put(self, key: str, request: Request, provider: str, response: Response) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"request": request.key_payload(provider), "response": {**asdict(response), "cached": False}}
        partial = path.with_suffix(".partial")
        partial.write_text(json.dumps(entry, indent=1, ensure_ascii=False))
        os.replace(partial, path)  # atomic: a crash never leaves a half-written entry
