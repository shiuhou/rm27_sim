"""Explicit asset locations for an independent simulator checkout."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSET_CONFIG = Path(os.environ.get('RM27_SIM_ASSET_CONFIG', ROOT / '.local/assets.json')).resolve()
PREFIXES = {
    'field': 'rm27/simulation/field/rmuc2026_battlefield_mjcf',
    'cad': 'data/references/cad_raw',
    'mjcf': 'components/rm_mjcf',
    'recordings': 'artifacts',
    'ardupilot': 'rm27/autopilot/vendor/ardupilot',
}


def resolve_asset(value):
    path = Path(value)
    if path.is_absolute():
        return path
    entries = json.loads(ASSET_CONFIG.read_text()) if ASSET_CONFIG.exists() else {}
    for name, prefix in PREFIXES.items():
        try:
            relative = path.relative_to(prefix)
        except ValueError:
            continue
        if name in entries:
            return Path(entries[name]).expanduser().resolve() / relative
    return ROOT / path


def configure(entries):
    current = json.loads(ASSET_CONFIG.read_text()) if ASSET_CONFIG.exists() else {}
    for name, value in entries.items():
        if value is not None:
            path = Path(value).expanduser().resolve()
            if not path.is_dir():
                raise ValueError(f'{name} asset directory not found: {path}')
            current[name] = str(path)
    ASSET_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    ASSET_CONFIG.write_text(json.dumps(current, indent=2)+'\n')
    return current
