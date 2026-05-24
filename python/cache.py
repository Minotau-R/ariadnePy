from pathlib import Path
import os
import re
import tempfile
import zipfile
import shutil

import pandas as pd
try:
    from platformdirs import user_cache_dir
except ImportError:
    try:
        from appdirs import user_cache_dir
    except ImportError:
        user_cache_dir = None

try:
    import requests
except ImportError:
    requests = None

try:
    import pyreadr
except ImportError:
    pyreadr = None


def init_cache() -> Path:
    """Initialize the ariadne resource cache directory."""
    if user_cache_dir is not None:
        cache_dir = Path(user_cache_dir("ariadne")) / "resource_cache"
    else:
        if os.name == "nt":
            base_dir = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        else:
            base_dir = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
        cache_dir = base_dir / "ariadne" / "resource_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def add_to_cache(df: pd.DataFrame, cache_path) -> str:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return str(cache_path)


def _download_file(url: str, dest_path: Path) -> Path:
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if requests is None:
        import urllib.request

        with urllib.request.urlopen(url) as response, dest_path.open("wb") as fh:
            shutil.copyfileobj(response, fh)
        return dest_path

    response = requests.get(url, stream=True)
    response.raise_for_status()
    with dest_path.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                fh.write(chunk)
    return dest_path


def _localize_resource(url: str) -> Path:
    path = Path(url)
    if path.exists():
        return path
    if str(url).startswith(("http://", "https://")):
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=Path(url).suffix or "")
        tmp_file.close()
        return _download_file(url, Path(tmp_file.name))
    raise FileNotFoundError(f"Resource not found: {url}")


def process_one2one(path, from_col, to_col):
    path = Path(path)
    df = pd.read_csv(path, sep="\t", header=None, dtype=str)
    if df.shape[1] < 2:
        raise ValueError("Expected at least two columns for one-to-one processing")
    first = df.iloc[:, 0].astype(str).str.replace(r"^.*?:", "", regex=True)
    second = df.iloc[:, 1].astype(str).str.replace(r"^GO:", "", regex=True)
    return pd.DataFrame({from_col: first, to_col: second})


def process_one2many(path, from_col, to_col, key_fn=None, val_fn=None, skiprows=0, val_cols=None):
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    if skiprows:
        lines = lines[skiprows:]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        key = parts[0]
        if val_cols is None:
            selection = parts[1:]
        elif isinstance(val_cols, slice):
            selection = parts[val_cols]
        else:
            selection = [parts[i] for i in val_cols if 0 <= i < len(parts)]
        for val in selection:
            if val is None or val == "":
                continue
            transformed_key = key_fn(key) if key_fn else key
            transformed_val = val_fn(val) if val_fn else val
            rows.append({from_col: transformed_key, to_col: transformed_val})
    return pd.DataFrame(rows, columns=[from_col, to_col])


def process_complex_modules(path, from_col, to_col):
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            key = parts[0]
            values = parts[1:]
            for value in values:
                for token in re.split(r"[;,]", value):
                    token = token.strip()
                    if not token:
                        continue
                    rows.append({from_col: key, to_col: token})
    return pd.DataFrame(rows, columns=[from_col, to_col])


def _extract_zip(path: Path, dest_dir: Path):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(dest_dir)
    extracted = []
    queue = [dest_dir]
    while queue:
        root = queue.pop()
        for entry in root.iterdir():
            if entry.is_dir():
                queue.append(entry)
                continue
            extracted.append(entry)
            if zipfile.is_zipfile(entry):
                inner_dir = entry.with_suffix("")
                inner_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(entry, "r") as zf:
                    zf.extractall(inner_dir)
                queue.append(inner_dir)
    return extracted


def _load_rds_files(paths):
    if pyreadr is None:
        raise ImportError("pyreadr is required to read RDS files")
    data_frames = []
    for path in paths:
        records = pyreadr.read_r(path)
        for obj in records.values():
            if isinstance(obj, pd.DataFrame):
                data_frames.append(obj)
            elif isinstance(obj, pd.Series):
                data_frames.append(obj.to_frame().T.reset_index(drop=True))
            elif isinstance(obj, dict):
                converted = {
                    key: pd.Series(value) if not isinstance(value, pd.Series) else value
                    for key, value in obj.items()
                }
                data_frames.append(pd.DataFrame(converted))
            elif isinstance(obj, (list, tuple)):
                data_frames.append(pd.DataFrame(obj))
    if not data_frames:
        raise ValueError("No RDS tables could be loaded from the archive")
    return pd.concat(data_frames, ignore_index=True, sort=False)


def process_rdslist(url):
    if str(url).startswith(("http://", "https://")):
        with tempfile.TemporaryDirectory() as tmpdir:
            local_zip = Path(tmpdir) / "resource.zip"
            _download_file(url, local_zip)
            extracted = _extract_zip(local_zip, Path(tmpdir) / "extracted")
    else:
        path = Path(url)
        if not path.exists() or path.suffix.lower() != ".zip":
            raise ValueError("MSigDB resource must be a zip archive path or URL")
        with tempfile.TemporaryDirectory() as tmpdir:
            extracted = _extract_zip(path, Path(tmpdir) / "extracted")
    rds_files = [p for p in extracted if p.suffix.lower() == ".rds" and "summary" not in p.name.lower()]
    if not rds_files:
        raise ValueError("No RDS files found in MSigDB archive")
    return _load_rds_files(rds_files)


def cache_resource(url, res_name, from_col, to_col):
    cache_dir = init_cache()
    basename = Path(url).name or res_name
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", basename)
    cache_path = cache_dir / f"{res_name}_{safe_name}.parquet"
    if cache_path.exists():
        return str(cache_path)
    source_path = _localize_resource(url)
    if res_name == "ChocoPhlAn":
        df = process_one2many(
            source_path,
            from_col,
            to_col,
            key_fn=lambda k: k.replace("GO:", ""),
        )
    elif res_name == "WoL":
        key_fn = (lambda k: "UniRef90_" + k) if from_col == "uniref90" else (lambda k: k)
        df = process_one2many(
            source_path,
            from_col,
            to_col,
            key_fn=key_fn,
            val_fn=lambda v: v.replace("EC-", ""),
        )
    elif res_name == "BugSigDB":
        def bug_sig_key(k):
            cleaned = re.sub(r"^bsdb:", "", k)
            return re.sub(r"_.*$", "", cleaned)

        df = process_one2many(
            source_path,
            from_col,
            to_col,
            key_fn=bug_sig_key,
            skiprows=1,
            val_cols=slice(2, None),
        )
    elif res_name in {"TIGRFAMs", "GO"}:
        df = process_one2one(source_path, from_col, to_col)
    elif res_name == "GM":
        df = process_complex_modules(source_path, from_col, to_col)
    elif res_name == "MSigDB":
        df = process_rdslist(url)
    else:
        df = process_one2many(source_path, from_col, to_col)
    return add_to_cache(df, cache_path)
