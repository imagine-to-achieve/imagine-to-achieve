"""Fail before allocating models if the audited experiment implementation drifted."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
manifest = ROOT / 'outputs/push_t_view_weight_ablation_fp32_h13_20260904/runtime_source_manifest.json'
expected = json.loads(manifest.read_text())
failures = [name for name, checksum in expected.items()
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != checksum]
if failures:
    raise RuntimeError('Audited ablation source/config changed: '+', '.join(failures))
print(f'Validated {len(expected)} experiment source/config identities.', flush=True)
