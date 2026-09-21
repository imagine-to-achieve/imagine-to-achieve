"""Durable local spooling for optional diagnostics; shared I/O is best effort."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        with temporary.open('wb') as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


class DiagnosticJsonlSpool:
    """Keep an in-memory and node-local copy before publishing a snapshot.

    Optional file I/O must never fail a rollout. An unsuccessful publication
    leaves the complete local copy available for retry and exit-time export.
    Each worker owns its filename; atomic replacement prevents partial JSONL.
    """
    def __init__(self, destination: Path, local_path: Path):
        self.destination = Path(destination)
        self.local_path = Path(local_path)
        self.lines: list[str] = []
        self.last_error: str | None = None
        self._warned: set[str] = set()

    def _write(self, path: Path, payload: bytes) -> bool:
        try:
            atomic_write(path, payload)
            self.last_error = None
            return True
        except OSError as error:
            self.last_error = f'{path}: {error}'
            key = f'{path}:{error.errno}'
            if key not in self._warned:
                print(f'[diagnostic-spool] Buffered optional diagnostics; {self.last_error}', flush=True)
                self._warned.add(key)
            return False

    def append(self, record: dict, *, publish: bool) -> bool:
        # Serialization errors indicate invalid data, not optional I/O failures.
        self.lines.append(json.dumps(record, allow_nan=False) + '\n')
        payload = ''.join(self.lines).encode('utf-8')
        local_ok = self._write(self.local_path, payload)
        if publish:
            return self._write(self.destination, payload)
        return local_ok

    def publish(self) -> bool:
        return self._write(self.destination, ''.join(self.lines).encode('utf-8'))


def make_view_reward_spool(run_dir: Path, rank: int) -> DiagnosticJsonlSpool:
    job = os.environ.get('SLURM_JOB_ID', 'local')
    filename = f'rank_{rank:03d}_job_{job}.jsonl'
    scratch_root = Path('/scratch/local') / job
    local_root = scratch_root if job.isdigit() and scratch_root.is_dir() else Path(tempfile.gettempdir())
    local_dir = local_root / 'reward_view_records' / run_dir.name
    return DiagnosticJsonlSpool(run_dir / 'reward_view_records' / filename, local_dir / filename)


def publish_directory(source: Path, destination: Path) -> int:
    failed = 0
    for path in sorted(source.glob('rank_*.jsonl')):
        try:
            atomic_write(destination / path.name, path.read_bytes())
        except OSError as error:
            print(f'[diagnostic-spool] Final publication failed for {path}: {error}', flush=True)
            failed += 1
    return failed


def probe_directory(directory: Path, workers: int = 4) -> dict:
    """Check concurrent create/reopen/append/fsync on the actual shared path."""
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.io_probe_', dir=directory) as temporary:
        def probe(index):
            path = Path(temporary) / f'rank_{index}.jsonl'
            for _ in range(3):
                with path.open('ab') as handle:
                    handle.write(b'x' * 65535 + b'\n')
                    handle.flush()
                    os.fsync(handle.fileno())
            return path.stat().st_size
        with ThreadPoolExecutor(max_workers=workers) as pool:
            sizes = list(pool.map(probe, range(workers)))
    return {'host': os.uname().nodename, 'workers': workers, 'bytes_verified': sum(sizes)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--probe-dir', type=Path)
    args = parser.parse_args()
    if args.probe_dir is not None:
        print(json.dumps(probe_directory(args.probe_dir)), flush=True)
    elif args.source is not None and args.destination is not None:
        raise SystemExit(bool(publish_directory(args.source, args.destination)))
    else:
        parser.error('provide --probe-dir or both --source and --destination')
