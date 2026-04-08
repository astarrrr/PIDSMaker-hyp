from collections import deque
from types import SimpleNamespace

import torch

from pidsmaker.detection.training_methods.training_loop import compute_temporal_loss_for_batch


class DummyModel:
    def __init__(self):
        self.calls = []

    def compute_temporal_contrastive_loss(self, current_state, previous_state):
        self.calls.append(("bi", current_state, previous_state))
        return torch.tensor(1.0)

    def compute_tri_temporal_contrastive_loss(
        self, previous_previous_state, previous_state, current_state
    ):
        self.calls.append(("tri", previous_previous_state, previous_state, current_state))
        return torch.tensor(2.0)


def test_bi_temporal_mode_uses_latest_previous_window():
    model = DummyModel()
    temporal_cfg = SimpleNamespace(mode="bi")
    temporal_state_queue = deque(["w1", "w2"], maxlen=2)

    loss = compute_temporal_loss_for_batch(
        model=model,
        temporal_cfg=temporal_cfg,
        temporal_state_queue=temporal_state_queue,
        temporal_state="w3",
        device=torch.device("cpu"),
    )

    assert loss.item() == 1.0
    assert model.calls == [("bi", "w3", "w2")]


def test_tri_temporal_mode_uses_three_window_context_when_available():
    model = DummyModel()
    temporal_cfg = SimpleNamespace(mode="tri")
    temporal_state_queue = deque(["w1", "w2"], maxlen=2)

    loss = compute_temporal_loss_for_batch(
        model=model,
        temporal_cfg=temporal_cfg,
        temporal_state_queue=temporal_state_queue,
        temporal_state="w3",
        device=torch.device("cpu"),
    )

    assert loss.item() == 2.0
    assert model.calls == [("tri", "w1", "w2", "w3")]


def test_tri_temporal_mode_falls_back_to_pairwise_when_only_one_history_window_exists():
    model = DummyModel()
    temporal_cfg = SimpleNamespace(mode="tri")
    temporal_state_queue = deque(["w1"], maxlen=2)

    loss = compute_temporal_loss_for_batch(
        model=model,
        temporal_cfg=temporal_cfg,
        temporal_state_queue=temporal_state_queue,
        temporal_state="w2",
        device=torch.device("cpu"),
    )

    assert loss.item() == 1.0
    assert model.calls == [("bi", "w2", "w1")]
