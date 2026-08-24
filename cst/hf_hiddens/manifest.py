"""Hub manifest JSON for uploaded hidden chunks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence



MANIFEST_VERSION = 2

# Running totals written during store; ignored when checking whether a resume
# is the same draw (seed / sample count / batching / generation settings).
FINGERPRINT_PROGRESS_KEYS = frozenset(
    {
        "cst_question_tokens",
        "cst_gen_tokens",
        "cst_train_tokens",
        "cst_train_samples",
    }
)

STORE_TRAIN_FINGERPRINT_REQUIRED = (
    "train_seed",
    "train_samples_per_epoch",
    "train_inference_batch_size",
)
STORE_EVAL_FINGERPRINT_REQUIRED = (
    "eval_task",
    "eval_seed",
    "eval_inference_batch_size",
    "eval_num_fewshot",
    "eval_limit",
)


def normalize_fingerprint_value(v: Any) -> Any:
    """Canonical form for fingerprint equality (None / int / bool / str)."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip()
        if s.lower() in ("", "null", "none"):
            return None
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        try:
            return int(s)
        except ValueError:
            return s
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, int):
        return int(v)
    return v


def fingerprint_identity(fp: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Fingerprint fields that must match for store resume, excluding running totals."""
    if not fp:
        return {}
    return {k: v for k, v in fp.items() if k not in FINGERPRINT_PROGRESS_KEYS}


def fingerprint_progress(fp: Optional[Dict[str, Any]]) -> Dict[str, int]:
    """Running CST token counters from a stored fingerprint (0 if missing)."""
    src = fp or {}
    out: Dict[str, int] = {}
    for k in FINGERPRINT_PROGRESS_KEYS:
        raw = src.get(k)
        try:
            out[k] = 0 if raw is None else int(raw)
        except (TypeError, ValueError):
            out[k] = 0
    return out


def fingerprint_conflicts(
    stored: Optional[Dict[str, Any]],
    current: Dict[str, Any],
    *,
    required_keys: Sequence[str],
) -> List[str]:
    """Human-readable mismatches that block skipping already-stored batches."""
    if not stored:
        return ["Hub folder has chunks but no sampling_fingerprint; cannot verify the same draw"]
    msgs: List[str] = []
    stored_id = fingerprint_identity(stored)
    current_id = fingerprint_identity(current)
    keys = set(required_keys) | (set(stored_id) & set(current_id))
    for k in sorted(keys):
        if k in required_keys and k not in stored:
            msgs.append(f"missing {k} in stored fingerprint")
            continue
        sv = normalize_fingerprint_value(stored.get(k))
        cv = normalize_fingerprint_value(current.get(k))
        if sv != cv:
            msgs.append(f"{k}: stored={stored.get(k)!r} vs current={current.get(k)!r}")
    return msgs


def store_fingerprint_required_keys(scope: str) -> tuple[str, ...]:
    s = str(scope).strip().lower()
    if s == "eval":
        return STORE_EVAL_FINGERPRINT_REQUIRED
    return STORE_TRAIN_FINGERPRINT_REQUIRED


@dataclass
class ChunkRecord:
    batch_idx: int
    filename: str
    bytes: int

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChunkRecord":
        return cls(
            batch_idx=int(d["batch_idx"]),
            filename=str(d["filename"]),
            bytes=int(d["bytes"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_idx": int(self.batch_idx),
            "filename": str(self.filename),
            "bytes": int(self.bytes),
        }


def _chunks_from_legacy_segments(d: Dict[str, Any]) -> tuple[List[ChunkRecord], Optional[Dict[str, Any]]]:
    """Flatten old segment-based manifests into a chunk list."""
    by_name: Dict[str, ChunkRecord] = {}
    fingerprint: Optional[Dict[str, Any]] = None
    for seg in d.get("segments") or []:
        if fingerprint is None and seg.get("sampling_fingerprint") is not None:
            fingerprint = dict(seg["sampling_fingerprint"])
        for c in seg.get("chunks") or []:
            rec = ChunkRecord.from_dict(c)
            by_name[str(rec.filename)] = rec
    chunks = sorted(by_name.values(), key=lambda c: int(c.batch_idx))
    return chunks, fingerprint


@dataclass
class HiddenManifest:
    schema_version: int
    scope: str
    name_token: str
    chunks: List[ChunkRecord] = field(default_factory=list)
    sampling_fingerprint: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(cls, *, scope: str, name_token: str) -> "HiddenManifest":
        return cls(
            schema_version=MANIFEST_VERSION,
            scope=str(scope),
            name_token=str(name_token),
            chunks=[],
            sampling_fingerprint=None,
            meta={},
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HiddenManifest":
        fingerprint = d.get("sampling_fingerprint")
        if "chunks" in d:
            chunks = [ChunkRecord.from_dict(c) for c in d.get("chunks") or []]
        else:
            chunks, legacy_fp = _chunks_from_legacy_segments(d)
            if fingerprint is None:
                fingerprint = legacy_fp
        chunks.sort(key=lambda c: int(c.batch_idx))
        return cls(
            schema_version=int(d.get("schema_version", MANIFEST_VERSION)),
            scope=str(d.get("scope", "train")),
            name_token=str(d["name_token"]),
            chunks=chunks,
            sampling_fingerprint=dict(fingerprint) if fingerprint is not None else None,
            meta=dict(d.get("meta") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "schema_version": int(self.schema_version),
            "scope": str(self.scope),
            "name_token": str(self.name_token),
            "chunks": [c.to_dict() for c in self.chunks],
            "meta": dict(self.meta),
        }
        if self.sampling_fingerprint is not None:
            out["sampling_fingerprint"] = dict(self.sampling_fingerprint)
        return out

    def upsert_chunk(self, rec: ChunkRecord) -> None:
        for i, c in enumerate(self.chunks):
            if int(c.batch_idx) == int(rec.batch_idx) or str(c.filename) == str(rec.filename):
                self.chunks[i] = rec
                self.chunks.sort(key=lambda x: int(x.batch_idx))
                return
        self.chunks.append(rec)
        self.chunks.sort(key=lambda c: int(c.batch_idx))

    def all_chunks_sorted(self) -> List[ChunkRecord]:
        """Chunks sorted by batch_idx."""
        return sorted(self.chunks, key=lambda c: int(c.batch_idx))

    def chunk_by_batch_idx(self, batch_idx: int) -> Optional[ChunkRecord]:
        for c in self.chunks:
            if int(c.batch_idx) == int(batch_idx):
                return c
        return None

    def chunks_from_batch_idx(self, start_batch_idx: int) -> List[ChunkRecord]:
        """Chunks with batch_idx at or after start_batch_idx."""
        return [c for c in self.all_chunks_sorted() if int(c.batch_idx) >= int(start_batch_idx)]

    def total_batch_count(self) -> int:
        """Highest batch index plus one, or zero if empty."""
        if not self.chunks:
            return 0
        return max(int(c.batch_idx) for c in self.chunks) + 1

    def sampling_fingerprint_from_manifest(self) -> Optional[Dict[str, Any]]:
        if self.sampling_fingerprint is not None:
            return dict(self.sampling_fingerprint)
        return None

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=0)

    @classmethod
    def load_json(cls, path: str | Path) -> "HiddenManifest":
        with Path(path).open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
