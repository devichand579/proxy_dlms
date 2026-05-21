"""Hub manifest JSON for uploaded hidden chunks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

MANIFEST_VERSION = 1


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


@dataclass
class SegmentRecord:
    segment: int
    prefix: str
    chunks: List[ChunkRecord] = field(default_factory=list)
    total_bytes: int = 0
    next_batch_idx: int = 0
    sampling_fingerprint: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SegmentRecord":
        chunks = [ChunkRecord.from_dict(c) for c in d.get("chunks") or []]
        return cls(
            segment=int(d["segment"]),
            prefix=str(d["prefix"]),
            chunks=chunks,
            total_bytes=int(d.get("total_bytes", sum(c.bytes for c in chunks))),
            next_batch_idx=int(d.get("next_batch_idx", 0)),
            sampling_fingerprint=d.get("sampling_fingerprint"),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "segment": int(self.segment),
            "prefix": str(self.prefix),
            "chunks": [c.to_dict() for c in self.chunks],
            "total_bytes": int(self.total_bytes),
            "next_batch_idx": int(self.next_batch_idx),
        }
        if self.sampling_fingerprint is not None:
            out["sampling_fingerprint"] = self.sampling_fingerprint
        return out


@dataclass
class HiddenManifest:
    schema_version: int
    scope: str
    name_token: str
    segments: List[SegmentRecord] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(cls, *, scope: str, name_token: str) -> "HiddenManifest":
        return cls(
            schema_version=MANIFEST_VERSION,
            scope=str(scope),
            name_token=str(name_token),
            segments=[],
            meta={},
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HiddenManifest":
        segs = [SegmentRecord.from_dict(s) for s in d.get("segments") or []]
        return cls(
            schema_version=int(d.get("schema_version", MANIFEST_VERSION)),
            scope=str(d.get("scope", "train")),
            name_token=str(d["name_token"]),
            segments=segs,
            meta=dict(d.get("meta") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "scope": str(self.scope),
            "name_token": str(self.name_token),
            "segments": [s.to_dict() for s in self.segments],
            "meta": dict(self.meta),
        }

    def find_segment(self, segment: int) -> Optional[SegmentRecord]:
        for s in self.segments:
            if int(s.segment) == int(segment):
                return s
        return None

    def upsert_segment(self, rec: SegmentRecord) -> None:
        for i, s in enumerate(self.segments):
            if int(s.segment) == int(rec.segment):
                self.segments[i] = rec
                return
        self.segments.append(rec)

    def all_chunks_sorted(self) -> List[ChunkRecord]:
        """All chunks, deduped by filename, sorted by batch_idx."""
        by_name: Dict[str, ChunkRecord] = {}
        for seg in self.segments:
            for ch in seg.chunks:
                by_name[str(ch.filename)] = ch
        return sorted(by_name.values(), key=lambda c: int(c.batch_idx))

    def chunks_from_batch_idx(self, start_batch_idx: int) -> List[ChunkRecord]:
        """Chunks with batch_idx at or after start_batch_idx."""
        return [c for c in self.all_chunks_sorted() if int(c.batch_idx) >= int(start_batch_idx)]

    def chunks_for_download(
        self,
        *,
        segment: int,
        start_batch_idx: int,
    ) -> List[ChunkRecord]:
        """Chunks in one segment from start_batch_idx onward."""
        seg = self.find_segment(segment)
        if seg is None:
            return []
        out = [c for c in seg.chunks if int(c.batch_idx) >= int(start_batch_idx)]
        out.sort(key=lambda c: int(c.batch_idx))
        return out

    def total_batch_count(self) -> int:
        """Highest batch index plus one, or zero if empty."""
        mx = -1
        for seg in self.segments:
            for ch in seg.chunks:
                mx = max(mx, int(ch.batch_idx))
        return mx + 1 if mx >= 0 else 0

    def sampling_fingerprint_from_manifest(self) -> Optional[Dict[str, Any]]:
        for seg in self.segments:
            if seg.sampling_fingerprint is not None:
                return dict(seg.sampling_fingerprint)
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
