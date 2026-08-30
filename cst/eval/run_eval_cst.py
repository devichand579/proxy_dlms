"""Per-file or directory evaluation of hidden ``.pt`` vs a CST weight file."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import torch

from load_model import infer_diffusion_token_ids, load_model
from cst.eval.logits_head import logits_from_prelogit_hidden
from cst.eval.metrics import (
    compute_l0,
    compute_nmse,
    compute_ntf_logits,
    cosine_fidelity,
    unmasked_masked_token_masks,
    temporal_coherence,
)
from cst.model.cst_bank import CSTBank
from cst.model.residency import CSTResidency
from cst.train.hidden_chunk import load_chunk_H
from cst.train.trainer import load_cst_bank_weights, load_shard_if_exists

logger = logging.getLogger("cst.eval")

_EVAL_EMPTY_CACHE_EVERY = 1
_EVAL_PROGRESS_LOG_EVERY = 20


def _pinned_to_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Stage a CPU slice through pinned memory before a device transfer.

    ``H`` (see ``load_chunk_H``) is memory-mapped, so a slice of it is
    pageable host memory. Transferring straight from pageable/mmap'd memory
    on this ROCm/MI300X build routes through the AMDGPU driver's SVM/HMM
    subsystem and hangs under sustained use — the same root cause fixed on
    the training side via ``_DeviceRows.prepare`` in ``cst/train/trainer.py``.
    Pinned memory uses the ordinary DMA path instead.
    """
    if device.type == "cuda" and t.device.type == "cpu":
        t = t.pin_memory()
    return t.to(device)


def detect_max_trained_step(ckpt_path: str | Path) -> Optional[int]:
    """Highest step with a saved shard in a sharded checkpoint dir.

    Steps beyond this were never trained (``_load_sharded`` leaves them at random
    init when their shard is missing) — callers must cap eval anchors to this so
    untrained modules never contribute to reconstructions. Returns ``None`` for a
    legacy single-file ``.pt`` checkpoint (always a full dump, no cap needed).
    """
    path = Path(ckpt_path)
    if not path.is_dir():
        return None
    steps: List[int] = []
    for p in path.glob("cst_*.pt"):
        try:
            steps.append(int(p.stem.split("_")[-1]))
        except ValueError:
            continue
    return max(steps) if steps else None


def load_eval_cst_bank(
    ckpt_path: str | Path,
    cfg_bank: Dict[str, Any],
    device: torch.device,
    *,
    w: int | None = None,
    gpu_pinned_gb: float = 0.0,
) -> tuple[CSTBank, Optional[int], Optional[CSTResidency]]:
    """Build a ``CSTBank`` sized for eval and load ``ckpt_path`` into it.

    CST modules are held in fp32 once loaded (``load_cst_bank_weights`` /
    ``load_shard_if_exists`` both upcast from the bf16 on-disk storage dtype),
    so eagerly materializing the full ``cfg_bank["max_steps"]`` bank is
    expensive — e.g. 512 steps at ``d=4096,k=4096,w=4`` is ~172GB, which alone
    can leave no headroom for the base model or NTF logits. When the
    checkpoint is a sharded dir with fewer trained steps than the configured
    max (the common case: only the steps actually touched by training data
    have shards), this builds the bank ``lazy=True`` and only materializes
    steps ``1..detect_max_trained_step(ckpt_path)`` — the only ones eval will
    ever touch once the caller also caps ``evaluate_chunk_file``'s
    ``max_anchor`` to the same value. ``max_steps`` itself is left at the
    configured value (not shrunk to the cap) so ``effective_window_size``
    keeps computing the same window sizes near the tail that training used.

    Returns ``(cst_bank, max_trained_step_or_None, residency_or_None)``.
    """
    full_max_steps = int(cfg_bank["max_steps"])
    ctor_kwargs = dict(
        d=int(cfg_bank["d_model"]),
        k=int(cfg_bank["k"]),
        w=int(cfg_bank["w"]) if w is None else int(w),
        max_on_gpu=int(cfg_bank.get("max_on_gpu", full_max_steps)),
        initial_threshold=float(cfg_bank.get("jumprelu_initial_threshold", 0.03)),
        bandwidth=float(cfg_bank.get("jumprelu_bandwidth", 1.0)),
    )
    cap = detect_max_trained_step(ckpt_path)
    if cap is not None and cap < full_max_steps:
        cst_bank = CSTBank(max_steps=full_max_steps, lazy=True, **ctor_kwargs)
        ckpt_dir = Path(ckpt_path)
        for t in range(1, cap + 1):
            mod = cst_bank.get_or_create(t, device)
            load_shard_if_exists(ckpt_dir, t, mod)
        return cst_bank, cap, None

    cst_bank = CSTBank(max_steps=full_max_steps, **ctor_kwargs)
    load_cst_bank_weights(ckpt_path, cst_bank)
    residency: Optional[CSTResidency] = None
    if float(gpu_pinned_gb) > 0.0:
        residency = CSTResidency(cst_bank, device, pinned_bytes=int(float(gpu_pinned_gb) * 1e9))
    else:
        cst_bank.to_device_all(device)
    return cst_bank, cap, residency


@torch.no_grad()
def evaluate_chunk_file(
    cst_bank: CSTBank,
    chunk_path: str | Path,
    *,
    w: int,
    device: torch.device,
    max_anchor: int | None = None,
    base_model: Optional[torch.nn.Module] = None,
    ntf_conf_low: float = 0.2,
    ntf_conf_high: float = 0.8,
    ntf_first_non_eos: bool = False,
    eos_id: Optional[int] = None,
    residency: Optional[CSTResidency] = None,
) -> Dict[str, Any]:
    """NMSE, L0, cos, TC on one chunk; NTF and masks if base_model is given.

    With ``residency``, steps are streamed in on the anchor sweep instead of the
    whole bank being resident: anchor ``a`` reads ``a-w .. a`` (the extra step is
    the temporal-coherence partner), and everything below that is released.
    """
    H, meta, segs = load_chunk_H(chunk_path)
    T, B, L, D = H.shape
    if T < w:
        return {"error": "T < w", "path": str(chunk_path), "T": T, "w": w}

    U_stack: Optional[torch.Tensor] = None
    M_stack: Optional[torch.Tensor] = None
    segment_source = "none"
    if segs and "unmasked" in segs and "diffusion_masked" in segs:
        Uu, Mm = segs["unmasked"], segs["diffusion_masked"]
        if Uu.shape == (T, B, L) and Mm.shape == (T, B, L):
            U_stack = Uu.to(device)
            M_stack = Mm.to(device)
            segment_source = "per_step"
        else:
            logger.warning(
                "token mask shape %s / %s != H %s; falling back to prompt_len if set",
                tuple(Uu.shape),
                tuple(Mm.shape),
                (T, B, L),
            )
    if U_stack is None and meta.get("prompt_len") is not None:
        try:
            u0, m0 = unmasked_masked_token_masks(
                B, L, int(meta["prompt_len"]), device=device
            )
            U_stack = u0.unsqueeze(0).expand(T, -1, -1).contiguous()
            M_stack = m0.unsqueeze(0).expand(T, -1, -1).contiguous()
            segment_source = "prompt_len"
        except (TypeError, ValueError):
            pass

    def row_um(row: int) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if U_stack is not None and M_stack is not None and 0 <= row < T:
            return U_stack[row], M_stack[row]
        return None, None

    nmse_list: List[float] = []
    nmse_u_list: List[float] = []
    nmse_m_list: List[float] = []
    l0_list: List[float] = []
    l0_u_list: List[float] = []
    l0_m_list: List[float] = []
    cos_list: List[float] = []
    cos_u_list: List[float] = []
    cos_m_list: List[float] = []
    tc_list: List[float] = []
    ntf_list: List[float] = []
    ntf_u_list: List[float] = []
    ntf_m_list: List[float] = []

    def _ok(x: float) -> bool:
        return not (isinstance(x, float) and math.isnan(x))

    # H has T rows (0-indexed, valid indices 0..T-1); the max valid anchor is
    # T-1, matching training's own anchor range (iter_anchors: range(w_eff,
    # n_frames-1+1), i.e. up to n_frames-1). max_anchor further caps this to
    # the highest step actually trained.
    hi = min(T - 1, max_anchor) if max_anchor is not None else T - 1
    for anchor in range(w, hi + 1):
        if residency is not None:
            residency.ensure([t for t in range(max(1, anchor - w), anchor + 1)])
            for t in range(max(1, anchor - w - 4), max(1, anchor - w)):
                residency.release(t)
        s_target = anchor  # same H row as training (H_full[anchor])
        tgt = _pinned_to_device(H[s_target], device)
        H_hat = torch.zeros_like(tgt)
        u_tgt, m_tgt = row_um(anchor)

        for enc_slot in range(w):
            enc_step = anchor - w + 1 + enc_slot
            if enc_step < 1:
                continue
            offset = s_target - enc_step + 1
            if offset < 1:
                continue
            cst = cst_bank.get(enc_step)
            if offset > cst.w:
                continue
            H_enc = _pinned_to_device(H[enc_step - 1], device)
            A = cst.encode(H_enc.to(dtype=cst.W_enc.weight.dtype))
            H_hat = H_hat + cst.decode(A, offset)
            l0_list.append(compute_l0(A))
            u_enc, m_enc = row_um(enc_step - 1)
            if u_enc is not None and m_enc is not None:
                v_u = compute_l0(A, u_enc)
                v_m = compute_l0(A, m_enc)
                if _ok(v_u):
                    l0_u_list.append(v_u)
                if _ok(v_m):
                    l0_m_list.append(v_m)

            if enc_step > 1:
                c_prev = cst_bank.get(enc_step - 1)
                A_prev = c_prev.encode(
                    _pinned_to_device(H[enc_step - 2], device).to(dtype=c_prev.W_enc.weight.dtype)
                )
                if A.shape == A_prev.shape:
                    tc_list.append(temporal_coherence(A_prev, A))

        tf = tgt.float()
        hf = H_hat.float()
        nmse_list.append(compute_nmse(tf, hf))
        cos_list.append(cosine_fidelity(tf, hf))
        if u_tgt is not None and m_tgt is not None:
            nmu = compute_nmse(tf, hf, u_tgt)
            nmm = compute_nmse(tf, hf, m_tgt)
            if _ok(nmu):
                nmse_u_list.append(nmu)
            if _ok(nmm):
                nmse_m_list.append(nmm)
            cu = cosine_fidelity(tf, hf, u_tgt)
            cm = cosine_fidelity(tf, hf, m_tgt)
            if _ok(cu):
                cos_u_list.append(cu)
            if _ok(cm):
                cos_m_list.append(cm)

        if base_model is not None:
            logits_t = logits_from_prelogit_hidden(base_model, tgt)
            logits_c = logits_from_prelogit_hidden(base_model, H_hat)
            ntf = compute_ntf_logits(
                logits_t,
                logits_c,
                conf_low=ntf_conf_low,
                conf_high=ntf_conf_high,
                ntf_first_non_eos=ntf_first_non_eos,
                eos_id=eos_id,
            )
            if not math.isnan(ntf):
                ntf_list.append(ntf)
            if u_tgt is not None and m_tgt is not None:
                ntu = compute_ntf_logits(
                    logits_t,
                    logits_c,
                    conf_low=ntf_conf_low,
                    conf_high=ntf_conf_high,
                    token_mask=u_tgt,
                    ntf_first_non_eos=ntf_first_non_eos,
                    eos_id=eos_id,
                )
                ntm = compute_ntf_logits(
                    logits_t,
                    logits_c,
                    conf_low=ntf_conf_low,
                    conf_high=ntf_conf_high,
                    token_mask=m_tgt,
                    ntf_first_non_eos=ntf_first_non_eos,
                    eos_id=eos_id,
                )
                if _ok(ntu):
                    ntf_u_list.append(ntu)
                if _ok(ntm):
                    ntf_m_list.append(ntm)
            del logits_t, logits_c

        # logits_t/logits_c are [B, L, vocab_size] (vocab_size can be 100K+),
        # recomputed fresh every anchor — on ROCm the caching allocator does
        # not always reclaim these promptly across iterations, so without an
        # explicit periodic empty_cache() a long file's NTF pass can OOM well
        # before the CST bank + base model's own (much smaller) footprint
        # would explain. Cheap insurance; harmless if unnecessary.
        if (
            (anchor - w) % _EVAL_EMPTY_CACHE_EVERY == 0
            and torch.cuda.is_available()
            and device.type == "cuda"
        ):
            torch.cuda.empty_cache()

        if (anchor - w) % _EVAL_PROGRESS_LOG_EVERY == 0:
            logger.info(
                "eval progress: %s anchor=%d/%d nmse=%.6f",
                Path(chunk_path).name,
                anchor - w + 1,
                hi - w + 1,
                nmse_list[-1] if nmse_list else float("nan"),
            )

    out: Dict[str, Any] = {
        "path": str(chunk_path),
        "meta": meta,
        "nmse_mean": sum(nmse_list) / max(len(nmse_list), 1),
        "l0_mean": sum(l0_list) / max(len(l0_list), 1),
        "cos_mean": sum(cos_list) / max(len(cos_list), 1),
        "tc_mean": sum(tc_list) / max(len(tc_list), 1),
        "n_anchors": len(nmse_list),
        "n_tc": len(tc_list),
        "segmentation": segment_source,
    }
    if ntf_list:
        out["ntf_mean"] = sum(ntf_list) / len(ntf_list)
    elif base_model is not None:
        out["ntf_mean"] = float("nan")
    if nmse_u_list:
        out["nmse_unmasked"] = sum(nmse_u_list) / len(nmse_u_list)
    if nmse_m_list:
        out["nmse_masked"] = sum(nmse_m_list) / len(nmse_m_list)
    if l0_u_list:
        out["l0_unmasked"] = sum(l0_u_list) / len(l0_u_list)
    if l0_m_list:
        out["l0_masked"] = sum(l0_m_list) / len(l0_m_list)
    if cos_u_list:
        out["cos_unmasked"] = sum(cos_u_list) / len(cos_u_list)
    if cos_m_list:
        out["cos_masked"] = sum(cos_m_list) / len(cos_m_list)
    if ntf_u_list:
        out["ntf_unmasked"] = sum(ntf_u_list) / len(ntf_u_list)
    if ntf_m_list:
        out["ntf_masked"] = sum(ntf_m_list) / len(ntf_m_list)
    return out


EVAL_AGG_SPLIT_KEYS = (
    "nmse_unmasked",
    "nmse_masked",
    "l0_unmasked",
    "l0_masked",
    "cos_unmasked",
    "cos_masked",
    "ntf_unmasked",
    "ntf_masked",
)


def new_eval_aggregator() -> Dict[str, List[float]]:
    """Empty metric lists for run-level means."""
    return {k: [] for k in ("nmse", "l0", "cos", "tc", "ntf", *EVAL_AGG_SPLIT_KEYS)}


def merge_eval_file_into_agg(agg: Dict[str, List[float]], m: Dict[str, Any]) -> None:
    """Append one file's means into ``agg``."""
    if "nmse_mean" in m:
        agg["nmse"].append(m["nmse_mean"])
        agg["l0"].append(m["l0_mean"])
        agg["cos"].append(m["cos_mean"])
        agg["tc"].append(m["tc_mean"])
    if "ntf_mean" in m and not (isinstance(m["ntf_mean"], float) and math.isnan(m["ntf_mean"])):
        agg["ntf"].append(m["ntf_mean"])
    for sk in EVAL_AGG_SPLIT_KEYS:
        v = m.get(sk)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            agg[sk].append(float(v))


def eval_summary_from_agg(agg: Dict[str, List[float]]) -> Dict[str, float]:
    """Mean of each list -> ``summary`` dict."""
    summary: Dict[str, float] = {
        "nmse": sum(agg["nmse"]) / max(len(agg["nmse"]), 1),
        "l0": sum(agg["l0"]) / max(len(agg["l0"]), 1),
        "ntf_cos": sum(agg["cos"]) / max(len(agg["cos"]), 1),
        "tc": sum(agg["tc"]) / max(len(agg["tc"]), 1),
    }
    if agg["ntf"]:
        summary["ntf"] = sum(agg["ntf"]) / len(agg["ntf"])
    for sk in EVAL_AGG_SPLIT_KEYS:
        if agg[sk]:
            summary[sk] = sum(agg[sk]) / len(agg[sk])
    return summary


def run_eval_on_dir(
    hidden_dir: Union[str, Path, Iterable[Union[str, Path]]],
    ckpt_path: str | Path,
    cfg_bank: Dict[str, Any],
    *,
    device: str = "cuda",
    w: int | None = None,
    model_path: str | None = None,
    ntf_conf_low: float = 0.2,
    ntf_conf_high: float = 0.8,
    ntf_first_non_eos: bool = False,
    eos_id: Optional[int] = None,
    gpu_pinned_gb: float = 0.0,
    cst_bank: Optional[CSTBank] = None,
    max_anchor: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate every hidden ``.pt`` file; return per-file list and mean ``summary``.

    ``hidden_dir`` is either a local directory (globbed for ``*.pt``) or any iterable
    of paths — e.g. a generator that downloads one Hub chunk at a time, so files never
    all sit on disk simultaneously.

    ``cst_bank``, when given, is used as-is (already loaded and placed on a device) and
    ``ckpt_path``/``cfg_bank``/``gpu_pinned_gb`` are ignored for bank construction — lets
    a multi-task caller load the bank once and reuse it across tasks. ``max_anchor`` caps
    every file's evaluated anchor range (pass the highest step actually trained — see
    ``detect_max_trained_step`` — so untrained, randomly-initialized CST modules never
    contribute to a reconstruction).

    ``gpu_pinned_gb > 0`` (only used when ``cst_bank`` is not given) streams the bank
    instead of placing all of it at once. Eval needs only parameters (no grads, no
    optimizer state), but that is still ~172 GB at ``max_steps=512, k=4096``; set this
    if the bank does not fit.
    """
    dev = torch.device(device)
    base_model: Optional[torch.nn.Module] = None
    eos_id_for_ntf: Optional[int] = eos_id
    if model_path:
        base_model = load_model(
            str(model_path),
            device=device,
            compile_model=False,
            capture_pre_logits=False,
        )
        base_model.eval()
        if eos_id_for_ntf is None and ntf_first_non_eos:
            _mid, eos_id_for_ntf, _eot = infer_diffusion_token_ids(str(model_path))

    residency: Optional[CSTResidency] = None
    if cst_bank is None:
        cst_bank, detected_cap, residency = load_eval_cst_bank(
            ckpt_path, cfg_bank, dev, w=w, gpu_pinned_gb=gpu_pinned_gb
        )
        if max_anchor is None:
            max_anchor = detected_cap

    if isinstance(hidden_dir, (str, Path)):
        paths: Iterable[Union[str, Path]] = sorted(Path(hidden_dir).glob("*.pt"))
        if not paths:
            logger.warning("no .pt files in %s", hidden_dir)
            return {"per_file": [], "summary": {}}
    else:
        paths = hidden_dir

    W = cst_bank.w
    agg: Dict[str, List[float]] = new_eval_aggregator()
    per_file: List[Dict[str, Any]] = []
    for p in paths:
        m = evaluate_chunk_file(
            cst_bank,
            p,
            w=W,
            device=dev,
            max_anchor=max_anchor,
            base_model=base_model,
            ntf_conf_low=ntf_conf_low,
            ntf_conf_high=ntf_conf_high,
            ntf_first_non_eos=ntf_first_non_eos,
            eos_id=eos_id_for_ntf,
            residency=residency,
        )
        per_file.append(m)
        merge_eval_file_into_agg(agg, m)
        logger.info(
            "eval chunk %s: nmse=%.6f l0=%.4f cos=%.4f tc=%.4f%s n_anchors=%d",
            Path(p).name,
            m.get("nmse_mean", float("nan")),
            m.get("l0_mean", float("nan")),
            m.get("cos_mean", float("nan")),
            m.get("tc_mean", float("nan")),
            f" ntf={m['ntf_mean']:.4f}" if "ntf_mean" in m else "",
            m.get("n_anchors", 0),
        )

    summary: Dict[str, float] = eval_summary_from_agg(agg)

    return {
        "per_file": per_file,
        "summary": summary,
    }
