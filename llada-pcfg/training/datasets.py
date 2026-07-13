"""PyTorch datasets for PCFG LLaDA training."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from pcfg_data import ARCH_LEN, make_train_example
from pcfg_metrics import make_val_example


class PCFGTrainDataset(Dataset):
    def __init__(self, token_seqs, vocab, arch_len=ARCH_LEN):
        self.token_seqs = token_seqs
        self.vocab = vocab
        self.arch_len = arch_len

    def __len__(self):
        return len(self.token_seqs)

    def __getitem__(self, idx):
        ex = make_train_example(self.token_seqs[idx], self.vocab, arch_len=self.arch_len)
        return {
            "input_ids": torch.from_numpy(ex.input_ids.copy()),
            "attention_mask": torch.from_numpy(ex.attention_mask.copy()),
            "labels": torch.from_numpy(ex.labels.copy()),
            "mask_t": torch.tensor(ex.mask_t, dtype=torch.float32),
            "n_content": torch.tensor(ex.n_content, dtype=torch.int64),
        }


class PCFGValDataset(Dataset):
    """Held-out eval with deterministic masking per example index."""

    def __init__(self, token_seqs, vocab, arch_len=ARCH_LEN, seed: int = 0):
        self.token_seqs = token_seqs
        self.vocab = vocab
        self.arch_len = arch_len
        self.seed = seed

    def __len__(self):
        return len(self.token_seqs)

    def __getitem__(self, idx):
        ex = make_val_example(self.token_seqs[idx], self.vocab, arch_len=self.arch_len, seed=self.seed + idx)
        return {
            "input_ids": torch.from_numpy(ex.input_ids.copy()),
            "attention_mask": torch.from_numpy(ex.attention_mask.copy()),
            "labels": torch.from_numpy(ex.labels.copy()),
            "mask_t": torch.tensor(ex.mask_t, dtype=torch.float32),
            "n_content": torch.tensor(ex.n_content, dtype=torch.int64),
        }
