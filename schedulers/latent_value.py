"""Amortized value of an actual greedy reveal for surviving-token predictions.

The training target is a reference log-likelihood change on disjoint MBPP data,
not mutual information or functional correctness. Deployment receives no labels.
"""
import torch

from schedulers.latent_risk import RiskHead, risk_features


def value_features(hidden, logits, embeddings, hp, ep, eos_id, masked_hidden):
    local = risk_features(hidden, logits, embeddings, hp, ep, eos_id)
    # Every candidate is in masked_hidden. Exclude its own representation.
    others = (masked_hidden.float().sum(0) - hidden.float()) / max(len(masked_hidden) - 1, 1)
    return torch.cat([local, others @ hp], -1)


class ValueReadout:
    def __init__(self, path, device="cuda"):
        a = torch.load(path, map_location=device, weights_only=True)
        self.hp, self.ep = a['h_projection'], a['e_projection']
        self.mean, self.scale = a['mean'], a['scale']
        self.feature_indices = a.get('feature_indices')
        self.head = RiskHead(len(self.mean)).to(device)
        self.head.load_state_dict(a['head']); self.head.eval()
        self.metadata = a['metadata']

    @torch.no_grad()
    def predict(self, hidden, logits, embeddings, eos_id, masked_hidden):
        x = value_features(hidden, logits, embeddings, self.hp, self.ep, eos_id, masked_hidden)
        if self.feature_indices is not None:
            x = x[:, self.feature_indices]
        # Standardized gain supplies a bounded relative-ranking term downstream.
        return self.head((x - self.mean) / self.scale)
