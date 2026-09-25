"""Portable, atomic local result files and deterministic random stream IDs."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def stable_seed(*parts):
    # This encoding is part of the published experiment's stream definition.
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:16], 'little')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_relative(path):
    value = Path(path)
    if value.is_absolute() or '..' in value.parts or not value.parts:
        raise ValueError('Use a nonempty relative path inside this replication folder')
    resolved = (ROOT / value).resolve()
    if ROOT not in resolved.parents:
        raise ValueError('Output must be inside the replication folder')
    return resolved


def json_value(value):
    if isinstance(value, dict):return {str(k): json_value(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):return json_value(value.tolist())
    if isinstance(value, np.generic):return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):return None
    if isinstance(value, Path):return str(value)
    return value


def write_json(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(json_value(value), indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_npz(path, **arrays):
    path=Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f, **arrays)
    tmp.replace(path)


def write_csv(path, rows):
    path=Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)
    tmp.replace(path)


def implementation_hashes():
    return {str(p.relative_to(ROOT)):digest(p)
            for p in sorted((ROOT/'replication').rglob('*')) if p.suffix in ('.py','.json')}
