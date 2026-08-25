"""Pre-logit hidden capture and .pt storage for CST."""

from __future__ import annotations

import errno
import json
import logging
import os
import queue
import struct
import threading
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from cst.storage_dtype import STORAGE_FLOAT_DTYPE, map_float_tensors

logger = logging.getLogger("cst.pre_logits")

PackFn = Callable[[torch.Tensor], torch.Tensor]

CST_HIDDEN_MAGIC = b"CSTHID01"
CST_HIDDEN_HEADER_RESERVE = 65536


class DiskFullError(OSError):
    """Local disk cannot hold the next hidden chunk; store should stop and resume later."""


def _header_blob(shape: Sequence[int], meta: Dict[str, Any]) -> bytes:
    raw = json.dumps(
        {"v": 1, "shape": [int(x) for x in shape], "dtype": "bfloat16", "meta": meta},
        separators=(",", ":"),
    ).encode("utf-8")
    blob = CST_HIDDEN_MAGIC + struct.pack("<Q", len(raw)) + raw
    if len(blob) > CST_HIDDEN_HEADER_RESERVE:
        raise ValueError(
            f"hidden header is {len(blob)} bytes; max {CST_HIDDEN_HEADER_RESERVE}"
        )
    return blob.ljust(CST_HIDDEN_HEADER_RESERVE, b"\x00")


def _raise_disk_full(exc: BaseException, path: Path) -> None:
    if isinstance(exc, DiskFullError):
        raise exc
    err = getattr(exc, "errno", None)
    if err == errno.ENOSPC or "no space left" in str(exc).lower():
        raise DiskFullError(f"disk full while writing {path}: {exc}") from exc
    raise exc


def is_cst_hidden_file(path: str | Path) -> bool:
    try:
        with Path(path).open("rb") as f:
            return f.read(8) == CST_HIDDEN_MAGIC
    except OSError:
        return False


def load_cst_hidden_tensor(
    path: str | Path,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Memory-map a CSTHID01 chunk as ``[T, B, L, D]`` bf16 plus meta."""
    path = Path(path)
    with path.open("rb") as f:
        magic = f.read(8)
        if magic != CST_HIDDEN_MAGIC:
            raise ValueError(f"{path}: not a CST hidden file")
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
    shape = tuple(int(x) for x in header["shape"])
    if len(shape) != 4:
        raise ValueError(f"{path}: shape must be [T, B, L, D], got {shape}")
    data_bytes = 2
    for s in shape:
        data_bytes *= int(s)
    nbytes = CST_HIDDEN_HEADER_RESERVE + data_bytes
    actual = path.stat().st_size
    if actual < nbytes:
        raise RuntimeError(f"{path}: file size {actual} < expected {nbytes}")
    storage = torch.UntypedStorage.from_file(str(path), shared=True, nbytes=nbytes)
    t = torch.tensor([], dtype=STORAGE_FLOAT_DTYPE)
    t.set_(storage, CST_HIDDEN_HEADER_RESERVE // 2, shape)
    return t, dict(header.get("meta") or {})


def write_cst_hidden_file(
    path: str | Path,
    tensor: torch.Tensor,
    meta: Dict[str, Any],
) -> None:
    """Write one ``[T, B, L, D]`` bf16 tensor as CSTHID01 (single sequential write)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h = tensor.detach()
    if h.dtype != STORAGE_FLOAT_DTYPE:
        h = map_float_tensors(h, to_storage=True)
    if h.device.type != "cpu":
        h = h.cpu()
    h = h.contiguous()
    if h.dim() != 4:
        raise ValueError(f"write_cst_hidden_file expects [T, B, L, D], got {tuple(h.shape)}")
    blob = _header_blob(h.shape, meta)
    try:
        with path.open("wb") as fh:
            fh.write(blob)
            h.view(torch.uint8).numpy().tofile(fh)
    except OSError as e:
        path.unlink(missing_ok=True)
        _raise_disk_full(e, path)


class HiddenStepSink:
    """Stream packed steps into the final chunk file; disk writes overlap the next forward."""

    def __init__(self, path: str | Path, *, async_write: bool = True) -> None:
        self.path = Path(path)
        self.n = 0
        self.shape_bld: Optional[Tuple[int, int, int]] = None
        self._fh: Optional[BinaryIO] = None
        self._q: Optional[queue.Queue] = None
        self._thread: Optional[threading.Thread] = None
        self._write_error: Optional[BaseException] = None
        self._async = bool(async_write)
        self._sentinel = object()

    def _ensure_file(self, bld: Tuple[int, int, int]) -> None:
        self.shape_bld = bld
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "wb", buffering=8 * 1024 * 1024)
        self._fh.write(b"\x00" * CST_HIDDEN_HEADER_RESERVE)
        if self._async:
            self._q = queue.Queue(maxsize=2)
            self._thread = threading.Thread(
                target=self._writer, name="cst-hidden-sink", daemon=True
            )
            self._thread.start()

    def _writer(self) -> None:
        assert self._q is not None
        while True:
            item = self._q.get()
            if item is self._sentinel:
                break
            if self._write_error is not None:
                continue
            try:
                self._write_step(item)
            except BaseException as e:
                self._write_error = e

    def _write_step(self, h: torch.Tensor) -> None:
        assert self._fh is not None
        try:
            h.view(torch.uint8).numpy().tofile(self._fh)
        except OSError as e:
            _raise_disk_full(e, self.path)

    def _raise_if_failed(self) -> None:
        err = self._write_error
        if err is not None:
            _raise_disk_full(err, self.path)

    def append(self, h: torch.Tensor) -> None:
        self._raise_if_failed()
        if h.dim() != 3:
            raise ValueError(f"HiddenStepSink expects [B, L, D], got {tuple(h.shape)}")
        h = h.detach()
        if h.dtype != STORAGE_FLOAT_DTYPE:
            h = h.to(STORAGE_FLOAT_DTYPE)
        if h.device.type != "cpu":
            h = h.cpu()
        h = h.contiguous()
        bld = (int(h.shape[0]), int(h.shape[1]), int(h.shape[2]))
        if self.shape_bld is None:
            self._ensure_file(bld)
        elif bld != self.shape_bld:
            raise ValueError(f"hidden step shape {bld} != first step {self.shape_bld}")
        if self._q is not None:
            while True:
                self._raise_if_failed()
                try:
                    self._q.put(h, timeout=0.5)
                    break
                except queue.Full:
                    continue
        else:
            self._write_step(h)
        self.n += 1

    def close(self) -> None:
        if self._q is not None:
            self._q.put(self._sentinel)
            if self._thread is not None:
                self._thread.join()
            self._q = None
            self._thread = None
        self._raise_if_failed()
        fh = self._fh
        if fh is not None:
            fh.flush()
            fh.close()
            self._fh = None

    def finalize(self, meta: Dict[str, Any]) -> None:
        """Patch the reserved header so the file is a loadable CSTHID01 chunk."""
        self.close()
        if self.shape_bld is None or self.n < 1:
            raise RuntimeError("HiddenStepSink.finalize() with no steps")
        b, l, d = self.shape_bld
        blob = _header_blob((self.n, b, l, d), meta)
        data_bytes = int(self.n) * b * l * d * 2
        expected = CST_HIDDEN_HEADER_RESERVE + data_bytes
        actual = self.path.stat().st_size
        if actual != expected:
            raise RuntimeError(
                f"hidden sink {self.path} size {actual} != expected {expected} "
                f"(T={self.n} B={b} L={l} D={d})"
            )
        with self.path.open("r+b") as fh:
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())

    def unlink(self) -> None:
        try:
            self.close()
        except Exception:
            pass
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass


def extract_pre_logits_hidden(outputs: Any) -> torch.Tensor:
    """Last layer from model hidden_states."""
    if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
        raise ValueError(
            "Model output has no hidden_states; call forward with output_hidden_states=True."
        )
    hs = outputs.hidden_states
    if len(hs) == 0:
        raise ValueError("hidden_states is empty.")
    return hs[-1]


def forward_with_pre_logits_hidden(
    model: nn.Module,
    *args: Any,
    **kwargs: Any,
) -> Tuple[Any, torch.Tensor]:
    """Forward pass; return outputs and last hidden as float32."""
    kwargs = dict(kwargs)
    kwargs["output_hidden_states"] = True
    kwargs["return_dict"] = True
    outputs = model(*args, **kwargs)
    h = extract_pre_logits_hidden(outputs).detach().float()
    return outputs, h


def _extract_input_ids_from_forward(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Optional[torch.Tensor]:
    if kwargs.get("input_ids") is not None:
        return kwargs["input_ids"]
    if args and isinstance(args[0], torch.Tensor):
        return args[0]
    return None


def _unwrap_lm(module: nn.Module) -> nn.Module:
    seen: set[int] = set()
    m: nn.Module = module
    while id(m) not in seen:
        seen.add(id(m))
        inner = getattr(m, "_inner", None)
        if isinstance(inner, nn.Module):
            m = inner
            continue
        if type(m).__name__ == "PreLogitsCaptureWrapper":
            wrapped = getattr(m, "model", None)
            if isinstance(wrapped, nn.Module):
                m = wrapped
                continue
        break
    return m


def find_final_norm(module: nn.Module) -> Optional[nn.Module]:
    """Final pre-logit norm (LLaDA ``transformer.ln_f``, Dream ``model.norm``)."""
    m = _unwrap_lm(module)
    transformer = None
    if hasattr(m, "model") and hasattr(m.model, "transformer"):
        transformer = m.model.transformer
    elif hasattr(m, "transformer"):
        transformer = m.transformer
    if transformer is not None:
        ln = getattr(transformer, "ln_f", None)
        if isinstance(ln, nn.Module):
            return ln
    base = getattr(m, "model", m)
    norm = getattr(base, "norm", None)
    if isinstance(norm, nn.Module):
        return norm
    return None


class PreLogitsCaptureWrapper(nn.Module):
    """Wrap the LM and record pre-logit hiddens each forward.

    Captures the final-norm output via a hook so generation does not materialize
    every transformer layer on GPU. Values are packed (optional) and stored as
    bfloat16, either in a RAM list or streamed to a ``HiddenStepSink``.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        enabled: bool = True,
        move_to_cpu: bool = True,
        capture_initial_embed: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.enabled = enabled
        self.move_to_cpu = move_to_cpu
        self.capture_initial_embed = capture_initial_embed
        self.records: List[torch.Tensor] = []
        self.pack_fn: Optional[PackFn] = None
        self.sink: Optional[HiddenStepSink] = None
        self._n_captured = 0
        self._pending: Optional[torch.Tensor] = None
        self._hook_handle: Optional[Any] = None
        self._use_all_hidden_states = True
        norm = find_final_norm(model)
        if norm is not None:
            self._hook_handle = norm.register_forward_hook(self._on_final_norm)
            self._use_all_hidden_states = False
        else:
            logger.warning(
                "PreLogitsCaptureWrapper: no final-norm module found; "
                "falling back to output_hidden_states=True (higher GPU memory)"
            )

    def configure_capture(
        self,
        *,
        pack_fn: Optional[PackFn] = None,
        sink: Optional[HiddenStepSink] = None,
    ) -> None:
        self.pack_fn = pack_fn
        self.sink = sink
        self.clear_records()

    def _on_final_norm(self, _module: nn.Module, _inputs: Any, output: Any) -> None:
        if not self.enabled:
            return
        t = output[0] if isinstance(output, (tuple, list)) else output
        if isinstance(t, torch.Tensor):
            self._pending = t

    def _embed_initial(self, input_ids: torch.Tensor) -> torch.Tensor:
        m = self.model
        if not hasattr(m, "get_input_embeddings"):
            raise RuntimeError("PreLogitsCaptureWrapper: inner model has no get_input_embeddings().")
        emb = m.get_input_embeddings()(input_ids)
        return emb.detach()

    def _ingest(self, h: torch.Tensor) -> None:
        if h.dim() != 3:
            return
        if self.pack_fn is not None:
            h = self.pack_fn(h)
        h = h.detach()
        if h.dtype != STORAGE_FLOAT_DTYPE:
            h = h.to(STORAGE_FLOAT_DTYPE)
        if self.sink is not None:
            self.sink.append(h)
        else:
            if self.move_to_cpu:
                h = h.cpu()
            self.records.append(h)
        self._n_captured += 1

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        kwargs = dict(kwargs)
        kwargs["return_dict"] = True
        kwargs["output_hidden_states"] = bool(self._use_all_hidden_states)
        input_ids = _extract_input_ids_from_forward(args, kwargs)
        self._pending = None
        if self.enabled and input_ids is not None and self.capture_initial_embed and self._n_captured == 0:
            self._ingest(self._embed_initial(input_ids))
        outputs = self.model(*args, **kwargs)
        if self.enabled:
            h: Optional[torch.Tensor] = self._pending
            self._pending = None
            if h is None and not self._use_all_hidden_states:
                try:
                    h = extract_pre_logits_hidden(outputs)
                except ValueError:
                    h = None
            elif h is None:
                h = extract_pre_logits_hidden(outputs)
            if h is not None:
                self._ingest(h)
        return outputs

    def finish_capture(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        return self.records

    def clear_records(self) -> None:
        self.records.clear()
        self._n_captured = 0
        self._pending = None

    def unwrap(self) -> nn.Module:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        return self.model

    def __getattr__(self, name: str) -> Any:
        if name in (
            "model",
            "enabled",
            "move_to_cpu",
            "records",
            "capture_initial_embed",
            "pack_fn",
            "sink",
        ):
            return super().__getattr__(name)
        return getattr(self.model, name)


def save_hidden_records(
    records: Union[torch.Tensor, Sequence[torch.Tensor]],
    path: Union[str, Path],
    *,
    prompt_len: int,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Save hidden_states as a CSTHID01 ``.pt`` (legacy torch.save files still load)."""
    meta: Dict[str, Any] = {"prompt_len": int(prompt_len)}
    if extra_meta:
        meta.update(extra_meta)
    if torch.is_tensor(records):
        hs = records
    else:
        hs = torch.stack(
            [map_float_tensors(t, to_storage=True) for t in records], dim=0
        )
    if hs.dim() != 4:
        raise ValueError(f"save_hidden_records expected [T, B, L, D], got {tuple(hs.shape)}")
    write_cst_hidden_file(path, hs, meta)
