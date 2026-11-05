import numpy as np
import torch

from fresco.hierarchy import hierarchy as h


def test_mask_and_abstain():
    p = [0, 0, 1, 1, 1]
    c = [0, 1, 2, 2, 0]  # (1, 0) occurs once -> noise
    m = h.build_mask(p, c, 2, 3, min_count=2)
    assert m.tolist() == [[0, 0, 0], [0, 0, 1]]
    try:  # child 0 appears under both parents
        h.build_mask(p, c, 2, 3, min_count=1)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    m = h.build_mask([0, 0, 1, 1], [0, 1, 2, 2], 2, 3)
    assert m.tolist() == [[1, 1, 0], [0, 0, 1]]
    # file matrix is (n_child, n_parent)
    f = [[1, 0], [1, 0], [0, 1]]
    assert h.build_mask(None, None, 2, 3, mapping=f).tolist() == m.tolist()
    ma = h.add_abstain(m)
    assert ma.shape == (3, 4) and ma[2, 3] == 1 and ma[2, :3].sum() == 0 and ma[:2, 3].sum() == 0


def test_decode_is_consistent():
    m = torch.tensor([[1.0, 1, 0], [0, 0, 1]])
    torch.manual_seed(0)
    lp, lc = torch.randn(500, 2) * 3, torch.randn(500, 3) * 3
    p, c = h.consistent_decode(lp, lc, m)
    assert h.inconsistency_rate(p, c, m) == 0.0
    # independent argmax is sometimes inconsistent on this input
    assert h.inconsistency_rate(lp.argmax(1), lc.argmax(1), m) > 0


def test_consistency_loss_zero_when_matching():
    m = torch.tensor([[1.0, 1, 0], [0, 0, 1]])
    lc = torch.randn(4, 3)
    marg = torch.softmax(lc, 1) @ m.t()
    assert h.consistency_loss(marg.log(), lc, m).abs() < 1e-5
    assert h.consistency_loss(torch.randn(4, 2), lc, m) >= 0
