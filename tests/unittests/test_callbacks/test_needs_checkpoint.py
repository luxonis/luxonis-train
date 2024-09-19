from luxonis_train.callbacks.needs_checkpoint import NeedsCheckpoint


def test_other_type():
    assert NeedsCheckpoint._get_other_type("loss") == "metric"
    assert NeedsCheckpoint._get_other_type("metric") == "loss"
