"""Download the pinned source files into data/raw/<source>/. Resumable; safe to re-run.

  python -m pipeline.download              # everything in config/sources.toml
  python -m pipeline.download nppes        # one source
"""

import hashlib
import json
import shutil
import subprocess
import sys
import time
import tomllib
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
SOURCES = tomllib.loads((ROOT / "config" / "sources.toml").read_text())
MAX_ATTEMPTS = 50  # CMS resets long connections; each attempt resumes from the bytes already saved
RETRY_DELAY_S = 15


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str) -> dict:
    url = SOURCES[name]["url"]
    out_dir = RAW / name
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / url.rsplit("/", 1)[1]
    done = target.with_name(target.name + ".done")
    if not done.exists():
        print(f"[{name}] downloading {url}", flush=True)
        # -C - resumes a partial file; --fail stops an error page being saved as data. Retried here rather than with
        # curl --retry, which doesn't cover a connection reset partway through a transfer.
        for attempt in range(1, MAX_ATTEMPTS + 1):
            result = subprocess.run(["curl", "-sS", "-L", "--fail", "-C", "-", "-o", str(target), url])
            if result.returncode == 0:
                break
            have = target.stat().st_size / 1e9 if target.exists() else 0
            print(f"[{name}] attempt {attempt} failed (curl {result.returncode}) at {have:.2f} GB; resuming", flush=True)
            time.sleep(RETRY_DELAY_S)
        else:
            raise RuntimeError(f"{name}: download failed after {MAX_ATTEMPTS} attempts")
        done.touch()
    if target.suffix == ".zip":
        with zipfile.ZipFile(target) as z:
            wanted = [n for n in z.namelist() if n.startswith("npidata_pfile") and "/" not in n]
            for member in wanted:
                if not (out_dir / member).exists():
                    print(f"[{name}] extracting {member}", flush=True)
                    # Extract to a temp name first, so an interrupted unzip can't later pass as a complete file.
                    partial = out_dir / (member + ".partial")
                    with z.open(member) as source, partial.open("wb") as sink:
                        shutil.copyfileobj(source, sink, 1 << 22)
                    partial.rename(out_dir / member)
    info = {"url": url, "file": target.name, "bytes": target.stat().st_size, "sha256": sha256(target)}
    print(f"[{name}] ready: {info['bytes'] / 1e9:.2f} GB", flush=True)
    return info


def main() -> None:
    names = sys.argv[1:] or list(SOURCES)
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        results = dict(zip(names, pool.map(fetch, names)))
    manifest_path = RAW / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest.update({k: {**v, "downloaded_at": stamp} for k, v in results.items()})
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
