"""A compact supervised readout of reference-disagreement risk from LLaDA internals."""
from __future__ import annotations

import torch


def risk_features(hidden, logits, token_embeddings, h_projection, e_projection, eos_id):
    """No reference labels enter this feature function or inference readout."""
    logz = logits.float().logsumexp(-1)
    values, indices = logits.topk(2, -1)
    logp = values.float() - logz[:, None]
    probabilities = logp.exp()
    tokens = indices[:, 0]
    scalar = torch.stack([-logp[:, 0], probabilities[:, 0], probabilities[:, 0] - probabilities[:, 1],
                          (tokens == eos_id).float()], -1)
    return torch.cat([hidden.float() @ h_projection, token_embeddings[tokens].float() @ e_projection, scalar], -1)


class RiskHead(torch.nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.network = torch.nn.Sequential(torch.nn.Linear(dimension, 64), torch.nn.SiLU(), torch.nn.Linear(64, 1))

    def forward(self, features):
        return self.network(features).squeeze(-1)


class RiskReadout:
    def __init__(self, path, device="cuda"):
        artifact = torch.load(path, map_location=device, weights_only=True)
        self.h_projection = artifact["h_projection"].to(device)
        self.e_projection = artifact["e_projection"].to(device)
        self.mean, self.scale = artifact["mean"].to(device), artifact["scale"].to(device)
        self.head = RiskHead(len(self.mean)).to(device)
        self.head.load_state_dict(artifact["head"])
        self.head.eval()
        self.metadata = artifact["metadata"]
        self.harmless = artifact.get("harmless")

    @torch.no_grad()
    def _features_and_probability(self, hidden, logits, embeddings, eos_id):
        features = risk_features(hidden, logits, embeddings, self.h_projection, self.e_projection, eos_id)
        if len(self.mean) == 4:
            features = features[:, -4:]
        probability = self.head((features - self.mean) / self.scale).sigmoid()
        return features, probability

    @torch.no_grad()
    def body_probability(self, hidden, logits, embeddings, eos_id):
        if self.metadata.get("feature_mode") != "body_gate":
            raise ValueError("content ranking requires a body-gate readout")
        return self._features_and_probability(hidden, logits, embeddings, eos_id)[1]

    @torch.no_grad()
    def predict(self, hidden, logits, embeddings, eos_id):
        features, probability = self._features_and_probability(hidden, logits, embeddings, eos_id)
        if self.metadata.get("feature_mode") == "body_gate":
            proposed = logits.argmax(-1)
            bad_tail = (~self.harmless[proposed]).float()
            return probability * (1 - features[:, -3]) + (1 - probability) * bad_tail
        return probability
