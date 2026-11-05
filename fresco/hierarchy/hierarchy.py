"""Utilities for using a parent/child task hierarchy.

The parent task (e.g. site) and child task (e.g. subsite) are linked by a
binary mask ``M`` of shape [n_parent, n_child] with ``M[p, c] = 1`` when child
class ``c`` belongs to parent class ``p``. When abstention is on, an abstain
row and column are appended (abstain <-> abstain only).
"""

import json
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


def build_mask(
    parent_labels,
    child_labels,
    n_parent: int,
    n_child: int,
    min_count: int = 1,
    mapping: Optional[Dict[int, list]] = None,
) -> np.ndarray:
    """Build the [n_parent, n_child] hierarchy mask from label ids.

    Args:
        parent_labels, child_labels: integer label ids (training split), same length.
        n_parent, n_child: number of classes WITHOUT the abstain class.
        min_count: a (parent, child) pair must co-occur at least this often
            to be valid; filters label noise. Ignored when ``mapping`` is given.
        mapping: optional [n_child, n_parent] 0/1 matrix (row = child, one 1 in
            the column of its parent) that overrides co-occurrence.
    """
    if mapping is not None:
        m = np.asarray(mapping, dtype=np.float32)
        if m.shape != (n_child, n_parent):
            raise ValueError(
                f"hierarchy mapping has shape {m.shape}, expected "
                f"({n_child}, {n_parent}) = (n_child, n_parent)"
            )
        if not np.isin(m, (0.0, 1.0)).all():
            raise ValueError("hierarchy mapping must contain only 0s and 1s")
        if not (m.sum(1) == 1).all():
            raise ValueError("each child (row) must have exactly one parent")
        mask = m.T.copy()  # internal mask is [n_parent, n_child]
    else:
        p = np.asarray(parent_labels, dtype=int)
        c = np.asarray(child_labels, dtype=int)
        counts = np.zeros((n_parent, n_child), dtype=np.int64)
        np.add.at(counts, (p, c), 1)
        mask = (counts >= min_count).astype(np.float32)
        multi = np.flatnonzero(mask.sum(0) > 1)
        if multi.size:
            raise ValueError(
                f"child classes {multi.tolist()} co-occur with more than one parent "
                f"(min_count={min_count}); each child must have exactly one parent. "
                "Raise min_count to drop noisy pairs or supply a mapping_file."
            )
    return mask


def load_mapping(path: str) -> np.ndarray:
    """Read a JSON [n_child, n_parent] 0/1 matrix; row c has a 1 in the column of its parent.

    Row/column order must match the label ids (id2label) of the two tasks.
    """
    with open(path, "r", encoding="utf-8") as f:
        return np.asarray(json.load(f), dtype=np.float32)


def add_abstain(mask: np.ndarray) -> np.ndarray:
    """Append an abstain row/column; abstain parent <-> abstain child only."""
    p, c = mask.shape
    out = np.zeros((p + 1, c + 1), dtype=mask.dtype)
    out[:p, :c] = mask
    out[p, c] = 1.0
    return out


def consistent_decode(logits_parent, logits_child, mask):
    """Jointly pick the best valid (parent, child) pair per sample.

    Returns (parent_pred, child_pred), each of shape [B].
    """
    joint = F.log_softmax(logits_parent, 1).unsqueeze(2) + F.log_softmax(
        logits_child, 1
    ).unsqueeze(1)
    joint = joint.masked_fill(mask.unsqueeze(0) == 0, float("-inf"))
    flat = joint.flatten(1).argmax(1)
    n_child = logits_child.shape[1]
    return flat // n_child, flat % n_child


def consistency_loss(logits_parent, logits_child, mask, reduction="mean"):
    """KL(p_parent || p_child @ M.T): child distribution marginalised to parents."""
    log_p1 = F.log_softmax(logits_parent, 1)
    p2 = F.softmax(logits_child, 1)
    marg = (p2 @ mask.t()).clamp_min(1e-8)  # [B, P]
    kl = (log_p1.exp() * (log_p1 - marg.log())).sum(1)
    return kl.mean() if reduction == "mean" else kl


def inconsistency_rate(parent_pred, child_pred, mask) -> float:
    """Fraction of predicted pairs with M[p, c] == 0."""
    m = torch.as_tensor(mask)
    p = torch.as_tensor(parent_pred, dtype=torch.long)
    c = torch.as_tensor(child_pred, dtype=torch.long)
    if p.numel() == 0:
        return 0.0
    return float((m[p, c] == 0).float().mean())


def _to_ids(series, label2id, unk_id):
    """Map a label column to ids; columns already integer-encoded pass through."""
    if np.issubdtype(series.dtype, np.integer):
        return series.to_numpy()
    return series.map(lambda v: label2id.get(str(v), unk_id)).to_numpy()


class Hierarchy:
    """Holds the hierarchy config and mask for the trainer and predictor."""

    def __init__(self, parent, child, mask, consistent_decoding=True, consistency_lambda=0.0):
        self.parent = parent
        self.child = child
        self.mask_np = np.asarray(mask, dtype=np.float32)
        self.consistent_decoding = consistent_decoding
        self.consistency_lambda = consistency_lambda
        self._mask_t = {}

    def mask(self, device):
        key = str(device)
        if key not in self._mask_t:
            self._mask_t[key] = torch.as_tensor(self.mask_np, device=device)
        return self._mask_t[key]

    def decode(self, logits):
        """Return {task: pred tensor} overrides for parent and child."""
        p, c = consistent_decode(
            logits[self.parent], logits[self.child], self.mask(logits[self.parent].device)
        )
        return {self.parent: p, self.child: c}

    def loss(self, logits):
        return self.consistency_lambda * consistency_loss(
            logits[self.parent], logits[self.child], self.mask(logits[self.parent].device)
        )

    @classmethod
    def from_data(cls, dw, kwargs, abstain=False):
        """Build from a DataHandler and the ``hierarchy_kwargs`` config block."""
        parent, child = kwargs["parent"], kwargs["child"]
        id2label = dw.dict_maps["id2label"]
        # abstain classes (if added) sit last; strip them to get the base sizes
        n_p = len(id2label[parent]) - (1 if abstain else 0)
        n_c = len(id2label[child]) - (1 if abstain else 0)
        p2id = {str(v): k for k, v in id2label[parent].items()}
        c2id = {str(v): k for k, v in id2label[child].items()}
        unk = dw.model_args["task_unks"]

        mapping = None
        if kwargs.get("mapping_file"):
            mapping = load_mapping(kwargs["mapping_file"])
        ytrain = dw.inference_data["y"]["train"]
        pl = _to_ids(ytrain[parent], p2id, p2id[unk[parent]])
        cl = _to_ids(ytrain[child], c2id, c2id[unk[child]])
        mask = build_mask(pl, cl, n_p, n_c, kwargs.get("min_count", 1), mapping)
        if abstain:
            mask = add_abstain(mask)
        return cls(
            parent,
            child,
            mask,
            kwargs.get("consistent_decoding", True),
            kwargs.get("consistency_lambda", 0.0),
        )
