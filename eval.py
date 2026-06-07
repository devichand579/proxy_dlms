"""lm-eval integration for LLaDA / Dream (SMDM-style patterns: github.com/ML-GSAI/SMDM)."""
import sys
import torch
import re
from pathlib import Path
import random
import numpy as np
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from load_model import (
    load_model_and_tokenizer,
    infer_diffusion_token_ids,
    is_dream_model_dir,
    prepend_bos_for_dream,
)
from schedulers.generate import generate
from traces.trace_wrapper import _task_trace_list_ctx
from traces.hf_trace_upload import maybe_upload_eval_trace_file

import warnings
import json
from collections import defaultdict
from dill._dill import PicklingWarning
import logging
import os
from datetime import datetime
import threading

# Length bucket boundaries (token count): prompts in the same bucket are batched together.
# Uses the upper bound of each bucket, e.g. (0, 32], (32, 64], (64, 128], ...
LENGTH_BUCKET_BOUNDARIES = [32, 64, 128, 256, 384, 512, 640, 768, 896, 1024, float("inf")]


def _bucket_for_length(length: int, boundaries=None):
    """Return the bucket key (upper bound) for a given prompt length."""
    if boundaries is None:
        boundaries = LENGTH_BUCKET_BOUNDARIES
    for b in boundaries:
        if length <= b:
            return b
    return boundaries[-1]

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

warnings.filterwarnings("ignore", category=PicklingWarning)


def _to_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes")
    return bool(val)


def setup_logger(name="llada_eval", log_dir="logs"):
    """File and console log; add timestamp to filename unless the name already has one."""
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create log filename - if name already looks like it has timestamp, use as-is
    # Otherwise add timestamp
    if re.match(r'.*_\d{8}_\d{6}$', name):
        # Name already has timestamp format (YYYYMMDD_HHMMSS)
        log_file = os.path.join(log_dir, f"{name}.log")
    else:
        # Add timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class StreamingTraceWriter:
    """JSONL or single JSON array output; thread-safe appends."""
    def __init__(self, filepath: str, task_name: str, resumable: bool = False):
        self._path = filepath
        self._task_name = task_name
        self._resumable = resumable
        self._count = 0
        self._lock = threading.Lock()

        if resumable:
            # JSONL: append mode, count existing lines for resume
            if os.path.exists(filepath):
                with open(filepath, "r") as rf:
                    for line in rf:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            self._count += 1
            self._f = open(filepath, "a")
            if self._count == 0:
                # Fresh file: write header line (comment for humans)
                self._f.write(f'# task={json.dumps(task_name)}\n')
                self._f.flush()
        else:
            self._f = open(filepath, "w")
            self._f.write('{"task": ')
            json.dump(task_name, self._f)
            self._f.write(', "samples": [')
            self._f.flush()

    def append(self, trace_dict: dict):
        with self._lock:
            if self._resumable:
                self._f.write(json.dumps(trace_dict, ensure_ascii=False) + "\n")
            else:
                if self._count > 0:
                    self._f.write(",")
                json.dump(trace_dict, self._f)
            self._count += 1
            self._f.flush()

    def close(self):
        with self._lock:
            if self._f is not None:
                if self._resumable:
                    # JSONL needs no footer
                    pass
                else:
                    self._f.write('], "num_samples": ')
                    self._f.write(str(self._count))
                    self._f.write("}")
                self._f.close()
                self._f = None

    @property
    def count(self):
        return self._count


def _load_checkpoint_completed(checkpoint_path: str) -> dict:
    """Read `checkpoint.jsonl`: map (task, doc) -> resps; last line wins per key."""
    completed = {}
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return completed
    try:
        with open(checkpoint_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                rec = json.loads(line)
                task = rec.get("task_name", "")
                doc_id = rec.get("doc_id", None)
                resps = rec.get("resps", [])
                if doc_id is not None and resps:
                    completed[(str(task), doc_id)] = resps
    except Exception as e:
        logging.getLogger("llada_eval").warning(f"Failed to load checkpoint {checkpoint_path}: {e}")
    return completed


def _diffusion_metrics_path(output_dir: str) -> str:
    return os.path.join(output_dir, ".eval_diffusion_metrics.json")


def _load_diffusion_metrics_aggregate(output_dir: str) -> dict:
    """Restore cumulative diffusion counters across process restarts (pairs with checkpoint.jsonl)."""
    path = _diffusion_metrics_path(output_dir)
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict):
            return {}
        out = {}
        for k in (
            "total_samples_global",
            "total_computed_samples_global",
            "total_nfe_mass_global",
            "total_steps_global",
            "total_blocks_global",
        ):
            if k in d and isinstance(d[k], int):
                out[k] = d[k]
        if "total_gen_time_global" in d and isinstance(d["total_gen_time_global"], (int, float)):
            out["total_gen_time_global"] = float(d["total_gen_time_global"])
        return out
    except (OSError, json.JSONDecodeError, TypeError) as e:
        logging.getLogger("llada_eval").warning(f"Could not load diffusion metrics {path}: {e}")
        return {}


def _atomic_write_json(path: str, payload: dict) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=0)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


@register_model("llada", "llada1_5", "dream")
class LLaDAEvalHarness(LM):
    """LLaDA/Dream harness for lm-eval: sampling, traces, and resumable checkpoints."""

    def __init__(
        self,
        model_path="./models/llada",
        mask_id=None,
        eos_id=None,
        eot_id=None,
        batch_size=8,
        mc_num=128,
        is_check_greedy=False,
        steps=256,
        gen_length=256,
        block_length=256,
        mode='pure_diffusion',
        strategy='confidence',
        top_k=1,
        compile_model=False,
        intervention_mode=None,
        intervention_steps=None,
        intervention_k=1,
        device="cuda",
        trace_name=None,
        task_name=None,
        resumable=False,
        output_dir=None,
        use_eos_harness: bool = True,
    ):
        '''
        Args:
            model_path: Local checkpoint directory (e.g. ./models/llada, ./models/llada1_5, ./models/dream).
            mask_id: [MASK] token id; default from config.json (mask_token_id).
            eos_id: EOS id for schedulers; default from config.
            eot_id: EoT id for logit suppression; from checkpoint (see load_model.infer_diffusion_token_ids) for model_type llada, else None (e.g. Dream).
            batch_size: mini batch size.
            mc_num: Monte Carlo estimation iterations
            is_check_greedy: For certain metrics like LAMBADA, the evaluation requires the model to verify whether the answer
                             is generated through greedy sampling conditioned on the prompt (note that this differs from conditional
                             generation). We implement this verification through the suffix_greedy_prediction() function, which
                             returns a True/False judgment used for accuracy calculation.
                             When is_check_greedy is set to True, the lm-evaluation-harness library automatically invokes this function.
                             However, since none of the metrics in the LLaDA paper (https://arxiv.org/abs/2502.09992) require this functionality,
                             we recommend setting is_check_greedy to False. This configuration causes suffix_greedy_prediction() to return False
                             by default, significantly accelerating the evaluation process.
            steps: Sampling steps, less than or equal to gen_length.
            gen_length: Generated answer length.
            block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive generation.
            mode: 'pure_diffusion' or 'block_diffusion' or 'top_k'.
            strategy: 'confidence' or 'random' or 'top_margin' or 'entropy'.
            top_k: Number of tokens to transfer per step (top_k scheduler only). Default 1.
            compile_model: Whether to compile model with torch.compile for faster inference
            intervention_mode: Optional intervention mode. "drop" to prevent tokens from being unmasked, "swap" to swap tokens.
            intervention_steps: Optional list/set of step indices (0-indexed) at which to apply intervention.
            trace_name: Optional trace name. If provided, trace will be saved to disk and uploaded to wandb as artifact.
            intervention_k: Number of tokens to drop/swap per intervention step. Default 1.
            use_eos_harness: If True (default), schedulers may return early on EOS and chunk decode truncates at first EOS.
                If False, full step budget and ``batch_decode`` of the full generated slice (reference LLaDA-style).
        '''
        super().__init__()

        self._model_path = str(model_path)
        self.device = torch.device(device)

        self.model, self.tokenizer = load_model_and_tokenizer(
            self._model_path,
            device=device,
            compile_model=compile_model,
            capture_pre_logits=False,
        )

        inf_mask, inf_eos, inf_eot = infer_diffusion_token_ids(self._model_path)
        self.mask_id = int(mask_id) if mask_id is not None else inf_mask
        self.eos_id = int(eos_id) if eos_id is not None else inf_eos
        self.eot_id = int(eot_id) if eot_id is not None else inf_eot

        self.model = self.model.to(self.device)
        self.model.eval()
        self._dream_prepend_bos = is_dream_model_dir(self._model_path)
        self.mode = str(mode)
        self.strategy = str(strategy)
        self.top_k = int(top_k)
        self.intervention_mode = str(intervention_mode) if intervention_mode is not None else None  # keep None when not set
        if intervention_steps is None:
            self.intervention_steps = None
        elif isinstance(intervention_steps, str):
            # Parse "0,128,256" or "0_128_256" from CLI
            self.intervention_steps = [int(x.strip()) for x in re.split(r'[,_]', intervention_steps) if x.strip()]
        else:
            self.intervention_steps = list(intervention_steps)
        self.intervention_k = int(intervention_k)
        if isinstance(use_eos_harness, str):
            self.use_eos_harness = use_eos_harness.strip().lower() in ("true", "1", "yes", "on")
        else:
            self.use_eos_harness = True if use_eos_harness is None else bool(use_eos_harness)
        self.trace_name = str(trace_name) if trace_name is not None else None
        self.task_name = task_name if task_name is not None else os.environ.get("TASK_NAME", "unknown")
        self.resumable = _to_bool(resumable) if resumable is not None else False
        self.output_dir = str(output_dir) if output_dir else os.environ.get("CHECKPOINT_DIR", "")
        # Resumable: persist wandb run id for ``launch_eval`` / ``--wandb_args resume=allow,id=...``
        # (W&B is often inited *after* the model, so this also runs in ``generate_until``.)
        self._maybe_persist_wandb_run_id()
        if self.tokenizer.padding_side != 'left':
            self.tokenizer.padding_side = 'left'

        assert self.tokenizer.pad_token_id != self.mask_id

        self.mc_num = int(mc_num)
        self.batch_size = int(batch_size)
        self.sampling_eps = 0.
        self.is_check_greedy = is_check_greedy

        self.steps = int(steps)
        self.gen_length = int(gen_length)
        if self.mode == "fhs":
            self.block_length = None
        else:
            self.block_length = int(block_length)

        # Global statistics tracking across all generate_until calls
        self.total_samples_global = 0
        # Samples that actually ran ``generate`` (excludes checkpoint-only rows in resumable mode)
        self.total_computed_samples_global = 0
        # sum_i steps_taken_i * n_i for correct sample-weighted avg NFE when steps vary per chunk
        self.total_nfe_mass_global = 0
        self.total_steps_global = 0
        self.total_gen_time_global = 0.0
        self.total_blocks_global = 0
        self.generate_until_call_count = 0
        # Lock for thread-safe updates to global statistics
        self._stats_lock = threading.Lock()

        # Set up logger
        self.logger = setup_logger(trace_name)

        if self.resumable and self.output_dir:
            _agg = _load_diffusion_metrics_aggregate(self.output_dir)
            if _agg:
                self.total_samples_global = int(_agg.get("total_samples_global", 0))
                self.total_computed_samples_global = int(_agg.get("total_computed_samples_global", 0))
                self.total_nfe_mass_global = int(_agg.get("total_nfe_mass_global", 0))
                self.total_steps_global = int(_agg.get("total_steps_global", 0))
                self.total_blocks_global = int(_agg.get("total_blocks_global", 0))
                self.total_gen_time_global = float(_agg.get("total_gen_time_global", 0.0))
                self.logger.info(
                    "Loaded cumulative diffusion metrics from checkpoint dir "
                    f"(samples={self.total_samples_global}, "
                    f"computed_samples={self.total_computed_samples_global}, "
                    f"nfe_mass={self.total_nfe_mass_global}, "
                    f"steps={self.total_steps_global})"
                )

        self.use_wandb = WANDB_AVAILABLE
        if self.use_wandb:
            try:
                if wandb.run is not None:
                    self.logger.info("Wandb already initialized by lm-eval-harness")
            except Exception as e:
                self.logger.warning(f"Wandb initialization check failed: {e}, disabling wandb logging")
                self.use_wandb = False
        else:
            self.logger.info("Wandb not available, skipping wandb logging")

        self.logger.info(f"Model path: {self._model_path}")
        if self._dream_prepend_bos:
            self.logger.info("Dream: prepending BOS to encoded prompts (upstream eval style)")
        self.logger.info(f"Mode: {self.mode}")
        self.logger.info(f"Strategy: {self.strategy}")
        self.logger.info(f"Top-k: {self.top_k}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"MC num: {self.mc_num}")
        self.logger.info(f"Steps: {self.steps}")
        self.logger.info(f"Gen length: {self.gen_length}")
        self.logger.info(f"Block length: {self.block_length}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Compile model: {compile_model}")
        self.logger.info(f"Mask id: {self.mask_id}")
        self.logger.info(f"EOS id: {self.eos_id}, EoT id: {self.eot_id}")
        self.logger.info(f"use_eos_harness: {self.use_eos_harness}")
        self.logger.info(f"Trace name: {self.trace_name}")
        self.logger.info(f"Intervention mode: {self.intervention_mode}")
        self.logger.info(f"Intervention steps: {self.intervention_steps}")
        self.logger.info(f"Intervention k: {self.intervention_k}")

    def _maybe_persist_wandb_run_id(self) -> None:
        """Write ``output_dir/.wandb_id`` on first success so a later process can pass ``resume=allow,id=...``."""
        if not self.resumable or not self.output_dir or not WANDB_AVAILABLE or wandb is None:
            return
        if wandb.run is None:
            return
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            wandb_id_file = os.path.join(self.output_dir, ".wandb_id")
            if not os.path.exists(wandb_id_file):
                with open(wandb_id_file, "w", encoding="utf-8") as f:
                    f.write(wandb.run.id)
        except OSError as e:
            log = logging.getLogger(__name__)
            log.warning("Could not write .wandb_id for W&B resume: %s", e)

    def _persist_diffusion_metrics_aggregate(self) -> None:
        """Write cumulative diffusion stats next to checkpoint (resumable mode).
        """
        if not self.resumable or not self.output_dir:
            return
        with self._stats_lock:
            payload = {
                "total_samples_global": int(self.total_samples_global),
                "total_computed_samples_global": int(self.total_computed_samples_global),
                "total_nfe_mass_global": int(self.total_nfe_mass_global),
                "total_steps_global": int(self.total_steps_global),
                "total_blocks_global": int(self.total_blocks_global),
                "total_gen_time_global": float(self.total_gen_time_global),
            }
        try:
            _atomic_write_json(_diffusion_metrics_path(self.output_dir), payload)
        except OSError as e:
            self.logger.warning("Could not persist diffusion metrics: %s", e)

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        logits = self.model(batch).logits
        return logits

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        raise NotImplementedError

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _maybe_bos_prompts(self, texts):
        """Dream checkpoints: prefix BOS before tokenize (see Dream eval ``add_bos_token``)."""
        if not self._dream_prepend_bos:
            return texts
        if isinstance(texts, str):
            return prepend_bos_for_dream(self.tokenizer, texts)
        return [prepend_bos_for_dream(self.tokenizer, t) for t in texts]

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        if self._dream_prepend_bos:
            context = prepend_bos_for_dream(self.tokenizer, context)

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests: list[Instance]):
        """Batched `generate_until`: length buckets, optional trace and resumable checkpoint."""
        # Increment call counter (thread-safe)
        with self._stats_lock:
            self.generate_until_call_count += 1

        total_requests = len(requests)
        if total_requests == 0:
            return []

        gen_cfg = requests[0].args[1]
        stop_tokens = gen_cfg.get("until", [])

        # When tracing is enabled, stream each sample's trace to disk (avoids holding all in memory)
        trace_writer = None
        checkpoint_writer = None
        if self.trace_name is not None:
            trace_base_dir = "trace_outputs"
            os.makedirs(trace_base_dir, exist_ok=True)
            trace_name_safe = re.sub(r'[^a-zA-Z0-9_.-]', '_', str(self.trace_name))
            trace_ext = "jsonl" if self.resumable else "json"
            task_trace_path = os.path.join(trace_base_dir, f"{trace_name_safe}.{trace_ext}")
            trace_writer = StreamingTraceWriter(
                task_trace_path, getattr(self, "task_name", "unknown"), resumable=self.resumable
            )
            _task_trace_list_ctx.set(trace_writer)

        # Resumable: load completed from checkpoint, open for appending new results
        completed_from_checkpoint = {}
        if self.resumable and self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            checkpoint_path = os.path.join(self.output_dir, "checkpoint.jsonl")
            completed_from_checkpoint = _load_checkpoint_completed(checkpoint_path)
            if completed_from_checkpoint:
                self.logger.info(f"Resuming: loaded {len(completed_from_checkpoint)} completed samples from checkpoint")
            checkpoint_writer = open(checkpoint_path, "a")
        # W&B inits after model construction; persist run id for next resume once the run is active
        self._maybe_persist_wandb_run_id()

        # Get prompt lengths and group request indices by length bucket for better batching
        question_texts = [req.args[0] for req in requests]
        question_texts = self._maybe_bos_prompts(question_texts)
        enc_no_pad = self.tokenizer(question_texts, add_special_tokens=False, padding=False)
        lengths = [len(ids) for ids in enc_no_pad["input_ids"]]
        bucket_to_indices = defaultdict(list)
        for i, L in enumerate(lengths):
            bucket_to_indices[_bucket_for_length(L)].append(i)
        for bucket_key in bucket_to_indices:
            bucket_to_indices[bucket_key].sort(key=lambda idx: lengths[idx])

        # Build chunks: each chunk is a list of request indices from the same bucket
        chunks = []
        for bucket_key in sorted(bucket_to_indices.keys(), key=lambda x: x if x != float("inf") else 1e9):
            indices = bucket_to_indices[bucket_key]
            for chunk_start in range(0, len(indices), self.batch_size):
                chunks.append(indices[chunk_start : chunk_start + self.batch_size])

        all_gen_texts = [None] * total_requests
        total_steps_taken = 0
        total_blocks_processed = 0
        total_gen_time = 0.0
        computed_samples_this_call = 0
        nfe_mass_this_call = 0  # sum of (steps_taken * batch_n) across chunks → sample-weighted NFE

        try:
            for chunk_index, chunk_indices in enumerate(chunks):
                computed_in_chunk = 0
                nfe_mass_chunk = 0
                steps_chunk = 0
                blocks_chunk = 0
                gen_time_chunk = 0.0

                chunk_requests = [requests[i] for i in chunk_indices]
                chunk_size = len(chunk_requests)

                # Resumable: split into completed (from checkpoint) and to_compute
                to_compute = []  # list of (orig_idx, req)
                for j, orig_idx in enumerate(chunk_indices):
                    req = chunk_requests[j]
                    task_name = getattr(req, "task_name", self.task_name) or self.task_name
                    doc_id = getattr(req, "doc_id", orig_idx)
                    key = (str(task_name), doc_id)
                    if self.resumable and key in completed_from_checkpoint:
                        resp = completed_from_checkpoint[key][0]
                        for stop in stop_tokens:
                            if stop in resp:
                                resp = resp.split(stop)[0]
                        all_gen_texts[orig_idx] = resp
                    else:
                        to_compute.append((orig_idx, req))

                # Run model only for to_compute
                if to_compute:
                    to_compute_indices, to_compute_requests = zip(*to_compute)
                    chunk_question_texts = self._maybe_bos_prompts(
                        [req.args[0] for req in to_compute_requests]
                    )

                    encoded_outputs = self.tokenizer(
                        chunk_question_texts,
                        add_special_tokens=False,
                        padding=True,
                        return_tensors="pt",
                    )
                    input_ids = encoded_outputs["input_ids"].to(self.device)
                    attention_mask = encoded_outputs["attention_mask"].to(self.device)

                    generated, steps_taken, blocks, gen_time = generate(
                        self.model,
                        input_ids,
                        attention_mask,
                        steps=self.steps,
                        gen_length=self.gen_length,
                        block_length=self.block_length,
                        mode=self.mode,
                        strategy=self.strategy,
                        top_k=self.top_k,
                        trace_name=self.trace_name,
                        upload_to_wandb=self.use_wandb,
                        intervention_mode=self.intervention_mode,
                        intervention_steps=self.intervention_steps,
                        intervention_k=self.intervention_k,
                        mask_id=self.mask_id,
                        eos_id=self.eos_id,
                        eot_id=self.eot_id,
                        use_eos_harness=self.use_eos_harness,
                        model_path=self._model_path,
                    )

                    prompt_len = int(input_ids.shape[1])
                    if not self.use_eos_harness:
                        chunk_gen_texts = self.tokenizer.batch_decode(
                            generated[:, prompt_len:], skip_special_tokens=True
                        )
                    else:
                        gen_tails = []
                        for b in range(generated.shape[0]):
                            tail = generated[b, prompt_len:]
                            eos_positions = (tail == self.eos_id).nonzero(as_tuple=False)
                            if eos_positions.numel() > 0:
                                tail = tail[: int(eos_positions[0].item())]
                            gen_tails.append(tail)
                        chunk_gen_texts = self.tokenizer.batch_decode(
                            gen_tails, skip_special_tokens=True
                        )
                    for i, gen_text in enumerate(chunk_gen_texts):
                        for stop in stop_tokens:
                            if stop in gen_text:
                                chunk_gen_texts[i] = gen_text.split(stop)[0]

                    for j, orig_idx in enumerate(to_compute_indices):
                        all_gen_texts[orig_idx] = chunk_gen_texts[j]
                        # Resumable: append newly computed sample to checkpoint
                        if checkpoint_writer is not None:
                            req = to_compute_requests[j]
                            try:
                                rec = {
                                    "doc_id": getattr(req, "doc_id", orig_idx),
                                    "task_name": getattr(req, "task_name", self.task_name),
                                    "prompt": req.args[0],
                                    "resps": [chunk_gen_texts[j]],
                                }
                                checkpoint_writer.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                checkpoint_writer.flush()
                            except Exception as e:
                                self.logger.warning(f"Failed to write checkpoint sample: {e}")

                    n_batch = len(to_compute)
                    computed_in_chunk = n_batch
                    steps_chunk = steps_taken
                    blocks_chunk = blocks
                    gen_time_chunk = gen_time
                    nfe_mass_chunk = steps_taken * n_batch

                    computed_samples_this_call += n_batch
                    total_steps_taken += steps_taken
                    total_blocks_processed += blocks
                    total_gen_time += gen_time
                    # steps_taken = number of batched forwards; each row in the batch runs every step ⇒
                    # per-sample NFE for this chunk == steps_taken
                    nfe_mass_this_call += nfe_mass_chunk

                    self.logger.info(
                        f"Processed chunk {chunk_index + 1}/{len(chunks)} (indices {chunk_indices}, "
                        f"computed={len(to_compute)}, skipped={chunk_size - len(to_compute)}, gen time: {gen_time:.4f}s)"
                    )
                    if self.use_wandb and wandb.run is not None:
                        try:
                            wandb.log({
                                "chunk/size": chunk_size,
                                "chunk/computed_batch_size": n_batch,
                                "chunk/gen_time": gen_time,
                                "chunk/steps_taken": steps_taken,
                                "chunk/nfe_per_sample": float(steps_taken),
                                "chunk/blocks_processed": blocks,
                                "chunk/throughput_samples_per_sec": chunk_size / gen_time if gen_time > 0 else 0,
                                "chunk/chunk_index": chunk_index,
                            })
                        except Exception as e:
                            self.logger.warning(f"Failed to log chunk metrics to wandb: {e}")
                else:
                    self.logger.info(
                        f"Processed chunk {chunk_index + 1}/{len(chunks)} (indices {chunk_indices}, "
                        f"all {chunk_size} from checkpoint, skipped)"
                    )

                # Keep diffusion aggregate on disk in step with flushed checkpoint JSONL (resumable).
                with self._stats_lock:
                    self.total_samples_global += chunk_size
                    self.total_computed_samples_global += computed_in_chunk
                    self.total_nfe_mass_global += nfe_mass_chunk
                    self.total_steps_global += steps_chunk
                    self.total_gen_time_global += gen_time_chunk
                    self.total_blocks_global += blocks_chunk
                if self.resumable and self.output_dir:
                    self._persist_diffusion_metrics_aggregate()

            # Close the trace file (footer already written by writer) and optionally upload artifact
            if trace_writer is not None:
                try:
                    trace_writer.close()
                    n = trace_writer.count
                    if n > 0:
                        self.logger.info(f"Wrote {n} trace(s) to {trace_writer._path}")
                        trace_name_safe = re.sub(r'[^a-zA-Z0-9_.-]', '_', str(self.trace_name))
                        if self.use_wandb and WANDB_AVAILABLE and wandb.run is not None:
                            try:
                                artifact = wandb.Artifact(name=f"trace_{trace_name_safe}", type="trace", description=f"All sample traces for task (trace_name={trace_name_safe})")
                                artifact.add_file(trace_writer._path, name=os.path.basename(trace_writer._path))
                                wandb.log_artifact(artifact)
                            except Exception as e:
                                self.logger.warning(f"Failed to upload task traces to wandb: {e}")
                        maybe_upload_eval_trace_file(
                            trace_writer._path,
                            trace_name=trace_name_safe,
                            task_name=getattr(self, "task_name", None),
                            num_samples=n,
                            log=self.logger,
                        )
                except Exception as e:
                    self.logger.warning(f"Failed to close trace file: {e}")

        finally:
            if trace_writer is not None:
                _task_trace_list_ctx.set(None)
            if checkpoint_writer is not None:
                try:
                    checkpoint_writer.close()
                except Exception as e:
                    self.logger.warning(f"Failed to close checkpoint: {e}")

        if total_requests > 0:
            nfe_avg_call = (
                nfe_mass_this_call / computed_samples_this_call if computed_samples_this_call > 0 else 0.0
            )
            skipped_ckpt_this_call = total_requests - computed_samples_this_call
            self.logger.info(f"Total requests in this call: {total_requests} (processed in chunks of {self.batch_size})")
            self.logger.info(
                f"Samples with diffusion this call: {computed_samples_this_call}"
                + (
                    ""
                    if skipped_ckpt_this_call == 0
                    else (
                        f" ({skipped_ckpt_this_call} served from checkpoint JSONL only — "
                        "NFE/step totals for those rows were accrued when they were first generated; "
                        "see cumulative globals / .eval_diffusion_metrics.json)"
                    )
                )
            )
            self.logger.info(f"Total steps taken in this call: {total_steps_taken}")
            self.logger.info(
                f"Avg NFE per sample this call: {nfe_avg_call:.4f}"
                " (= sum_chunk steps_taken_chunk × batch_size_chunk / sum batch sizes)"
            )
            self.logger.info(f"Total generation time in this call: {total_gen_time:.4f} seconds")
            self.logger.info(f"Total blocks in this call: {total_blocks_processed}")
            # Read global statistics within lock for consistency
            with self._stats_lock:
                call_count = self.generate_until_call_count
                total_samples = self.total_samples_global
                total_computed = self.total_computed_samples_global
                nfe_mass = self.total_nfe_mass_global
                total_steps = self.total_steps_global
                total_gen_time_global = self.total_gen_time_global
                total_blocks = self.total_blocks_global
            nfe_avg_global = nfe_mass / total_computed if total_computed > 0 else 0.0
            self.logger.info(f"--- Global Statistics (across {call_count} generate_until calls) ---")
            self.logger.info(f"Total samples (global): {total_samples}")
            self.logger.info(
                "Total computed samples (global, diffusion runs — includes prior runs "
                "if resuming from .eval_diffusion_metrics.json): "
                f"{total_computed}"
            )
            self.logger.info(f"Total steps taken (global): {total_steps}")
            self.logger.info(f"Avg NFE per sample (global): {nfe_avg_global:.4f}")
            self.logger.info(f"Total generation time (global): {total_gen_time_global:.4f} seconds")
            self.logger.info(f"Total blocks (global): {total_blocks}")
            self.logger.info(f"Number of generate_until calls: {call_count}")

            # Log global statistics to wandb
            if self.use_wandb and wandb.run is not None:
                try:
                    wandb.log({
                        "call/requests": total_requests,
                        "call/diffusion_samples": computed_samples_this_call,
                        "call/skipped_checkpoint_samples": skipped_ckpt_this_call,
                        "global/total_samples": total_samples,
                        "global/total_computed_samples": total_computed,
                        "global/total_steps": total_steps,
                        "global/total_nfe_mass": nfe_mass,
                        "global/total_gen_time": total_gen_time_global,
                        "global/total_blocks": total_blocks,
                        "global/generate_until_calls": call_count,
                        "global/avg_nfe_per_sample": nfe_avg_global,
                        # total_steps counts only diffusion forwards; per-request mean is diluted when
                        # some rows are checkpoint-only in the same generate_until.
                        "global/avg_steps_per_request_diluted": (
                            total_steps / total_samples if total_samples > 0 else 0
                        ),
                        # sum(steps_taken per chunk) / #diffusion samples; unlike nfe_mass avg if steps vary per chunk
                        "global/avg_steps_per_diffusion_chunk_sum": (
                            total_steps / total_computed if total_computed > 0 else 0
                        ),
                        "global/avg_gen_time_per_sample": total_gen_time_global / total_samples if total_samples > 0 else 0,
                        "global/throughput_samples_per_sec": total_samples / total_gen_time_global if total_gen_time_global > 0 else 0,
                    })
                except Exception as e:
                    self.logger.warning(f"Failed to log global metrics to wandb: {e}")

        torch.cuda.empty_cache()
        return all_gen_texts

if __name__ == "__main__":
    _repo_root = Path(__file__).resolve().parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    from repo_env_defaults import apply_repo_env_defaults

    apply_repo_env_defaults()
    set_seed(1234)
    # lm-eval-harness marks some tasks (e.g., Humaneval) as unsafe because they execute model-generated code.
    if "--confirm_run_unsafe_code" not in sys.argv:
        sys.argv.append("--confirm_run_unsafe_code")

    cli_evaluate()
