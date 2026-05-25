from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import List

import pandas as pd

from ariadnepy.exceptions import AriadneError, AriadneParseError

try:
    import pyreadr as _pyreadr
except ImportError:
    _pyreadr = None


def _extract_zip(zip_path: Path, dest_dir: Path) -> List[Path]:
    """Extract a zip archive, recursively unpacking nested zips."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    extracted: List[Path] = []
    queue = [dest_dir]
    while queue:
        root = queue.pop()
        for entry in root.iterdir():
            if entry.is_dir():
                queue.append(entry)
                continue
            extracted.append(entry)
            if zipfile.is_zipfile(entry):
                inner = entry.with_suffix("")
                inner.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(entry, "r") as zf:
                    zf.extractall(inner)
                queue.append(inner)
    return extracted


def _load_rds_files(paths: List[Path]) -> pd.DataFrame:
    """Read a list of RDS files and concatenate their DataFrames."""
    if _pyreadr is None:
        raise AriadneError(
            "pyreadr is required to read RDS files. "
            "Install with: pip install pyreadr"
        )
    frames: List[pd.DataFrame] = []
    for path in paths:
        records = _pyreadr.read_r(str(path))
        for obj in records.values():
            if isinstance(obj, pd.DataFrame):
                frames.append(obj)
            elif isinstance(obj, pd.Series):
                frames.append(obj.to_frame().T.reset_index(drop=True))
            elif isinstance(obj, dict):
                frames.append(pd.DataFrame(
                    {k: pd.Series(v) if not isinstance(v, pd.Series) else v
                     for k, v in obj.items()}
                ))
            elif isinstance(obj, (list, tuple)):
                frames.append(pd.DataFrame(obj))
    if not frames:
        raise AriadneParseError("No RDS tables could be loaded from the archive.")
    return pd.concat(frames, ignore_index=True, sort=False)


def process_rdslist(url: str) -> pd.DataFrame:
    """Download a zip of RDS files (MSigDB) and return a combined DataFrame."""
    try:
        import requests as _req
    except ImportError:
        _req = None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        if str(url).startswith(("http://", "https://")):
            if _req is None:
                import urllib.request
                local_zip = tmp / "resource.zip"
                urllib.request.urlretrieve(url, str(local_zip))
            else:
                resp = _req.get(url, stream=True, timeout=120)
                resp.raise_for_status()
                local_zip = tmp / "resource.zip"
                with open(local_zip, "wb") as fh:
                    for chunk in resp.iter_content(8192):
                        if chunk:
                            fh.write(chunk)
        else:
            local_zip = Path(url)
            if not local_zip.exists() or local_zip.suffix.lower() != ".zip":
                raise AriadneError("MSigDB resource must be a .zip file path or URL.")

        extracted = _extract_zip(local_zip, tmp / "extracted")

    rds_files = [
        p for p in extracted
        if p.suffix.lower() == ".rds" and "summary" not in p.name.lower()
    ]
    if not rds_files:
        raise AriadneParseError("No RDS files found in MSigDB archive.")
    return _load_rds_files(rds_files)
