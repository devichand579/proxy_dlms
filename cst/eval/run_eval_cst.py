"""Per-file or directory evaluation of hidden ``.pt`` vs a CST weight file."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

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
from cst.train.hidden_chunk import load_chunk_H
from cst.train.trainer import load_cst_bank_weights

logger = logging.getLogger("cst.eval")


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
) -> Dict[str, Any]:
    """NMSE, L0, cos, TC on one chunk; NTF and masks if base_model is given."""
    H, meta, segs = load_chunk_H(chunk_path)
    H = H.to(device)
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

    hi = min(T, max_anchor) if max_anchor is not None else T
    for anchor in range(w, hi + 1):
        s_target = anchor - 1  # same H row as training s_step
        tgt = H[s_target]
        H_hat = torch.zeros_like(tgt)
        u_tgt, m_tgt = row_um(anchor - 1)

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
            H_enc = H[enc_step - 1]
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
                A_prev = c_prev.encode(H[enc_step - 2].to(dtype=c_prev.W_enc.weight.dtype))
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
                logits_t, logits_c, conf_low=ntf_conf_low, conf_high=ntf_conf_high
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
                )
                ntm = compute_ntf_logits(
                    logits_t,
                    logits_c,
                    conf_low=ntf_conf_low,
                    conf_high=ntf_conf_high,
                    token_mask=m_tgt,
                )
                if _ok(ntu):
                    ntf_u_list.append(ntu)
                if _ok(ntm):
                    ntf_m_list.append(ntm)

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
    """Mean of each list → ``summary`` dict."""
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
    hidden_dir: str | Path,
    ckpt_path: str | Path,
    cfg_bank: Dict[str, Any],
    *,
    device: str = "cuda",
    w: int | None = None,
    model_path: str | None = None,
    ntf_conf_low: float = 0.2,
    ntf_conf_high: float = 0.8,
) -> Dict[str, Any]:
    """Evaluate all ``*.pt`` under ``hidden_dir``; return per-file list and mean ``summary``."""
    dev = torch.device(device)
    base_model: Optional[torch.nn.Module] = None
    if model_path:
        from load_model import load_model

        base_model = load_model(
            str(model_path),
            device=device,
            compile_model=False,
            capture_pre_logits=False,
        )
        base_model.eval()
    cst_bank = CSTBank(
        max_steps=int(cfg_bank["max_steps"]),
        d=int(cfg_bank["d_model"]),
        k=int(cfg_bank["k"]),
        w=int(cfg_bank["w"]) if w is None else int(w),
        max_on_gpu=int(cfg_bank.get("max_on_gpu", cfg_bank["max_steps"])),
        initial_threshold=float(cfg_bank.get("jumprelu_initial_threshold", 0.03)),
        bandwidth=float(cfg_bank.get("jumprelu_bandwidth", 1.0)),
    )
    load_cst_bank_weights(ckpt_path, cst_bank)
    cst_bank.to_device_all(dev)

    paths = sorted(Path(hidden_dir).glob("*.pt"))
    if not paths:
        logger.warning("no .pt files in %s", hidden_dir)
        return {"per_file": [], "summary": {}}

    W = cst_bank.w
    agg: Dict[str, List[float]] = new_eval_aggregator()
    per_file: List[Dict[str, Any]] = []
    for p in paths:
        m = evaluate_chunk_file(
            cst_bank,
            p,
            w=W,
            device=dev,
            base_model=base_model,
            ntf_conf_low=ntf_conf_low,
            ntf_conf_high=ntf_conf_high,
        )
        per_file.append(m)
        merge_eval_file_into_agg(agg, m)

    summary: Dict[str, float] = eval_summary_from_agg(agg)

    return {
        "per_file": per_file,
        "summary": summary,
    }
