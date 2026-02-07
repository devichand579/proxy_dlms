'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
'''
import torch
import re
from pathlib import Path
import random
import numpy as np
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from load_model import load_model_and_tokenizer
from schedulers.generate import generate

import warnings
from dill._dill import PicklingWarning
import time
import logging
import os
from datetime import datetime
import threading

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

warnings.filterwarnings("ignore", category=PicklingWarning)


def setup_logger(name="llada_eval", log_dir="logs"):
    """
    Set up a logger that writes to both file and console.
    Creates logs directory if it doesn't exist.
    If name already contains timestamp, use it as-is; otherwise add timestamp.
    """
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


@register_model("llada")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path="./models/llada",
        mask_id=126336,
        batch_size=8,
        mc_num=128,
        is_check_greedy=False,
        steps=256,
        gen_length=256,
        block_length=256,
        mode='pure_diffusion',
        strategy='confidence',
        compile_model=False,
        intervention_mode=None,
        intervention_steps=None,
        intervention_k=1,
        device="cuda",
        task_name=None,
    ):
        '''
        Args:
            model_path: LLaDA-8B-Base model path.
            mask_id: The token id of [MASK] is 126336.
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
            compile_model: Whether to compile model with torch.compile for faster inference
            intervention_mode: Optional intervention mode. "drop" to prevent tokens from being unmasked, "swap" to swap tokens.
            intervention_steps: Optional list/set of step indices (0-indexed) at which to apply intervention.
            intervention_k: Number of tokens to drop/swap per intervention step. Default 1.
        '''
        super().__init__()

        self.device = torch.device(device)
        self.model, self.tokenizer = load_model_and_tokenizer(model_path, device=device, compile_model=compile_model)

        self.model = self.model.to(self.device)
        self.model.eval()
        self.mode = str(mode)
        self.strategy = str(strategy)
        self.intervention_mode = str(intervention_mode)
        self.intervention_steps = list(intervention_steps) if intervention_steps is not None else None
        self.intervention_k = int(intervention_k)
        
        self.mask_id = int(mask_id)
        if self.tokenizer.padding_side != 'left':
            self.tokenizer.padding_side = 'left'

        assert self.tokenizer.pad_token_id != self.mask_id

        self.mc_num = int(mc_num)
        self.batch_size = int(batch_size)
        assert self.mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.is_check_greedy = is_check_greedy

        self.steps = int(steps)
        self.gen_length = int(gen_length)
        self.block_length = int(block_length)
        
        # Global statistics tracking across all generate_until calls
        self.total_samples_global = 0
        self.total_steps_global = 0
        self.total_gen_time_global = 0.0
        self.total_blocks_global = 0
        self.generate_until_call_count = 0
        # Lock for thread-safe updates to global statistics
        self._stats_lock = threading.Lock()
        
        # Extract model name from model_path (e.g., "llada" from "./models/llada")
        model_name = Path(model_path).name if model_path else "unknown"
        
        # Get task name from parameter or environment variable
        if task_name is None:
            task_name = os.environ.get("TASK_NAME", "unknown")
        
        # Sanitize names for filesystem (replace special chars with underscores)
        def sanitize_name(name):
            return re.sub(r'[^a-zA-Z0-9_-]', '_', str(name))
        
        model_name = sanitize_name(model_name)
        task_name = sanitize_name(task_name)
        
        # Create logger name: model_task_mode_timestamp
        # Mode: use short form (pd for pure_diffusion, bd for block_diffusion)
        mode_short = "pd" if mode == "pure_diffusion" else ("bd" if mode == "block_diffusion" else mode[:2])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger_name = f"{model_name}_{task_name}_{mode_short}_{timestamp}"
        
        # Set up logger
        self.logger = setup_logger(logger_name)
        
        # Initialize wandb if available (check if already initialized by lm-eval)
        self.use_wandb = WANDB_AVAILABLE
        if self.use_wandb:
            try:
                # Check if wandb is already initialized (by lm-eval-harness)
                if wandb.run is None:
                    # Not initialized yet, will be initialized by lm-eval-harness
                    self.use_wandb = True
                else:
                    # Already initialized
                    self.use_wandb = True
                    self.logger.info("Wandb already initialized by lm-eval-harness")
            except Exception as e:
                self.logger.warning(f"Wandb initialization check failed: {e}, disabling wandb logging")
                self.use_wandb = False
        else:
            self.logger.info("Wandb not available, skipping wandb logging")

        self.logger.info(f"Model path: {model_path}")
        self.logger.info(f"Mode: {self.mode}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"MC num: {self.mc_num}")
        self.logger.info(f"Steps: {self.steps}")
        self.logger.info(f"Gen length: {self.gen_length}")
        self.logger.info(f"Block length: {self.block_length}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Compile model: {compile_model}")
        self.logger.info(f"Mask id: {self.mask_id}")
    
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
        # seq = torch.concatenate([prefix, target])[None, :]
        # seq = seq.repeat((self.batch_size, 1)).to(self.device)

        # prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        # loss_acc = []
        # for _ in range(self.mc_num // self.batch_size):
        #     perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

        #     mask_indices = perturbed_seq == self.mask_id

        #     logits = self.get_logits(perturbed_seq, prompt_index)

        #     loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
        #     loss = loss.sum() / self.batch_size
        #     loss_acc.append(loss.item())

        # return - sum(loss_acc) / len(loss_acc)
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

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        # def _tokenize(e):
        #     prefix, target = self._encode_pair(e["prefix"], e["target"])
        #     return {
        #         "prefix_text": e["prefix"],
        #         "target_text": e["target"],
        #         "prefix": prefix,
        #         "target": target,
        #     }

        # ds = []
        # ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        # ds = Dataset.from_list(ds)
        # ds = ds.map(_tokenize)
        # ds = ds.with_format("torch")
        # prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        # assert max(prompt_len) <= 4096

        # out = []
        # with torch.no_grad():
        #     for elem in tqdm(ds, desc="Computing likelihood..."):
        #         prefix = elem["prefix"]
        #         target = elem["target"]

        #         ll = self.get_loglikelihood(prefix, target)

        #         is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

        #         out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        # torch.cuda.empty_cache()
        # return out
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests: list[Instance]):
        """
        Generate text for multiple requests using batched processing.
        Splits requests into chunks based on batch_size and processes them sequentially.
        """
        # Increment call counter (thread-safe)
        with self._stats_lock:
            self.generate_until_call_count += 1
        
        total_requests = len(requests)
        if total_requests == 0:
            return []
        
        gen_cfg = requests[0].args[1]
        stop_tokens = gen_cfg.get("until", [])
        
        all_gen_texts = []
        total_steps_taken = 0
        total_blocks_processed = 0
        total_gen_time = 0.0
        
        # Split requests into chunks based on batch_size
        for chunk_start in range(0, total_requests, self.batch_size):
            chunk_end = min(chunk_start + self.batch_size, total_requests)
            chunk_requests = requests[chunk_start:chunk_end]
            chunk_size = len(chunk_requests)
            
            # Extract question texts for this chunk
            chunk_question_texts = [req.args[0] for req in chunk_requests]
            
            # Tokenize chunk
            encoded_outputs = self.tokenizer(
                chunk_question_texts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt"
            )
            input_ids = encoded_outputs['input_ids'].to(self.device)
            attention_mask = encoded_outputs['attention_mask'].to(self.device)
            
            # Process chunk
            generated, steps_taken, blocks, gen_time = generate(
                self.model, 
                input_ids, 
                attention_mask, 
                steps=self.steps,
                gen_length=self.gen_length,
                block_length=self.block_length,
                mode=self.mode,
                strategy=self.strategy
            )
            
            # Extract generated parts (excluding prompts) and batch decode
            chunk_gen_texts = self.tokenizer.batch_decode(generated[:, input_ids.shape[1]:], skip_special_tokens=True)
            
            # Apply stop tokens
            for i, gen_text in enumerate(chunk_gen_texts):
                for stop in stop_tokens:
                    if stop in gen_text:
                        chunk_gen_texts[i] = gen_text.split(stop)[0]
            
            all_gen_texts.extend(chunk_gen_texts)
            total_steps_taken += steps_taken
            total_blocks_processed += blocks
            total_gen_time += gen_time
            
            self.logger.info(f"Processed chunk: {chunk_start}-{chunk_end-1} of {total_requests} requests (chunk size: {chunk_size}, gen time: {gen_time:.4f}s)")
            
            # Log chunk metrics to wandb
            if self.use_wandb and wandb.run is not None:
                try:
                    wandb.log({
                        "chunk/size": chunk_size,
                        "chunk/gen_time": gen_time,
                        "chunk/steps_taken": steps_taken,
                        "chunk/blocks_processed": blocks,
                        "chunk/throughput_samples_per_sec": chunk_size / gen_time if gen_time > 0 else 0,
                        "chunk/chunk_index": chunk_start // self.batch_size,
                    })
                except Exception as e:
                    self.logger.warning(f"Failed to log chunk metrics to wandb: {e}")
        
        # Update global statistics (thread-safe)
        with self._stats_lock:
            self.total_samples_global += total_requests
            self.total_steps_global += total_steps_taken
            self.total_gen_time_global += total_gen_time
            self.total_blocks_global += total_blocks_processed
        
        if total_requests > 0:
            self.logger.info(f"Total requests in this call: {total_requests} (processed in chunks of {self.batch_size})")
            self.logger.info(f"Total steps taken in this call: {total_steps_taken}")
            self.logger.info(f"Total generation time in this call: {total_gen_time:.4f} seconds")
            self.logger.info(f"Total blocks in this call: {total_blocks_processed}")
            # Read global statistics within lock for consistency
            with self._stats_lock:
                call_count = self.generate_until_call_count
                total_samples = self.total_samples_global
                total_steps = self.total_steps_global
                total_gen_time_global = self.total_gen_time_global
                total_blocks = self.total_blocks_global
            self.logger.info(f"--- Global Statistics (across {call_count} generate_until calls) ---")
            self.logger.info(f"Total samples (global): {total_samples}")
            self.logger.info(f"Total steps taken (global): {total_steps}")
            self.logger.info(f"Total generation time (global): {total_gen_time_global:.4f} seconds")
            self.logger.info(f"Total blocks (global): {total_blocks}")
            self.logger.info(f"Number of generate_until calls: {call_count}")
            
            # Log global statistics to wandb
            if self.use_wandb and wandb.run is not None:
                try:
                    wandb.log({
                        "global/total_samples": total_samples,
                        "global/total_steps": total_steps,
                        "global/total_gen_time": total_gen_time_global,
                        "global/total_blocks": total_blocks,
                        "global/generate_until_calls": call_count,
                        "global/avg_steps_per_sample": total_steps / total_samples if total_samples > 0 else 0,
                        "global/avg_gen_time_per_sample": total_gen_time_global / total_samples if total_samples > 0 else 0,
                        "global/throughput_samples_per_sec": total_samples / total_gen_time_global if total_gen_time_global > 0 else 0,
                    })
                except Exception as e:
                    self.logger.warning(f"Failed to log global metrics to wandb: {e}")
        
        torch.cuda.empty_cache()
        return all_gen_texts

if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
    