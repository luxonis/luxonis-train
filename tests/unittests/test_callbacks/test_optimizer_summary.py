"""Tests for the optimizer / parameter-group summary builder.

Two denominators are in play, chosen so that percentages sum naturally
in the axis the reader cares about:

- Group-level ``*_pct_of_model`` sums to 100% across all groups of all
  optimizers (modulo unclaimed / external parameters).
- Owner-level ``*_pct_of_owner`` sums to 100% across every appearance
  of a single owner in the summary — telling the reader how a node's
  parameters were split across groups.
"""

import io
from collections import defaultdict
from unittest.mock import patch

import torch
from loguru import logger
from rich.console import Console
from torch import nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LambdaLR

from luxonis_train.callbacks.luxonis_progress_bar import (
    build_optimizer_summary,
    log_optimizer_summary,
)


def _tiny_module(
    in_f: int, out_f: int, requires_grad: bool = True
) -> nn.Linear:
    m = nn.Linear(in_f, out_f)
    for p in m.parameters():
        p.requires_grad = requires_grad
    return m


def _numel(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _n_tensors(module: nn.Module) -> int:
    return sum(1 for _ in module.parameters())


def _owner_pct_totals(
    summary: dict,
) -> tuple[dict[str, float], dict[str, float]]:
    """Aggregate per-owner percentages across every group and
    optimizer.
    """
    tensors: dict[str, float] = defaultdict(float)
    params: dict[str, float] = defaultdict(float)
    for opt in summary["optimizers"]:
        for group in opt["groups"]:
            for owner in group["owners"]:
                tensors[owner["name"]] += owner["tensors_pct_of_owner"]
                params[owner["name"]] += owner["params_pct_of_owner"]
    return tensors, params


def test_single_optimizer_single_group_covers_100_percent():
    backbone = _tiny_module(4, 8)
    head = _tiny_module(8, 2)
    modules = {"backbone": backbone, "head": head}

    opt = SGD(
        [*backbone.parameters(), *head.parameters()], lr=0.01, momentum=0.9
    )
    summary = build_optimizer_summary([opt], [ConstantLR(opt)], modules)

    assert summary["n_optimizers"] == 1
    assert summary["trainable_tensors"] == _n_tensors(backbone) + _n_tensors(
        head
    )
    assert summary["trainable_params"] == _numel(backbone) + _numel(head)
    assert summary["frozen_tensors"] == 0
    assert summary["frozen_params"] == 0

    (opt_info,) = summary["optimizers"]
    assert opt_info["optimizer"] == "SGD"
    assert opt_info["scheduler"] == "ConstantLR"
    assert opt_info["n_groups"] == 1

    (group,) = opt_info["groups"]
    assert group["n_tensors"] == summary["trainable_tensors"]
    assert group["n_params"] == summary["trainable_params"]
    # Whole group covers 100% of trainable model params.
    assert group["tensors_pct_of_model"] == 100.0
    assert group["params_pct_of_model"] == 100.0

    # Every owner sits entirely in this one group → 100% of itself.
    for o in group["owners"]:
        assert o["tensors_pct_of_owner"] == 100.0
        assert o["params_pct_of_owner"] == 100.0


def test_owner_percentages_sum_to_100_when_split_across_groups():
    """The whole point of using an owner-relative denominator."""
    backbone = _tiny_module(4, 8)
    head = _tiny_module(8, 2)
    modules = {"backbone": backbone, "head": head}

    # Split BACKBONE across two groups (first tensor vs the rest),
    # keep head in a single group.
    bb_params = list(backbone.parameters())
    hd_params = list(head.parameters())
    opt = SGD(
        [
            {"params": bb_params[:1], "lr": 0.01},
            {"params": bb_params[1:], "lr": 0.02},
            {"params": hd_params, "lr": 0.001},
        ],
        momentum=0.9,
    )
    summary = build_optimizer_summary(
        [opt], [LambdaLR(opt, lr_lambda=lambda _e: 1.0)], modules
    )

    tensor_totals, param_totals = _owner_pct_totals(summary)
    assert set(tensor_totals) == {"backbone", "head"}
    for name in ("backbone", "head"):
        assert abs(tensor_totals[name] - 100.0) < 1e-9
        assert abs(param_totals[name] - 100.0) < 1e-9


def test_owner_percentages_sum_to_100_across_multiple_optimizers():
    """Multiple optimizers, one owner appears in each — still 100%
    total.
    """
    backbone = _tiny_module(4, 8)
    head = _tiny_module(8, 2)
    modules = {"backbone": backbone, "head": head}

    opt_backbone = SGD(list(backbone.parameters()), lr=0.01)
    opt_head = Adam(list(head.parameters()), lr=0.001)

    summary = build_optimizer_summary(
        [opt_backbone, opt_head],
        [ConstantLR(opt_backbone), CosineAnnealingLR(opt_head, T_max=10)],
        modules,
    )

    tensor_totals, param_totals = _owner_pct_totals(summary)
    for name in ("backbone", "head"):
        assert abs(tensor_totals[name] - 100.0) < 1e-9
        assert abs(param_totals[name] - 100.0) < 1e-9

    # And the group-level "% of model" still adds up to 100% globally.
    total_group_pct = sum(
        group["params_pct_of_model"]
        for opt in summary["optimizers"]
        for group in opt["groups"]
    )
    assert abs(total_group_pct - 100.0) < 1e-9


def test_frozen_params_appear_in_frozen_totals_not_in_percentages():
    trainable = _tiny_module(4, 8, requires_grad=True)
    frozen = _tiny_module(8, 2, requires_grad=False)
    modules = {"trainable": trainable, "frozen": frozen}

    opt = SGD([p for p in trainable.parameters() if p.requires_grad], lr=0.01)
    summary = build_optimizer_summary([opt], [ConstantLR(opt)], modules)

    assert summary["trainable_tensors"] == _n_tensors(trainable)
    assert summary["trainable_params"] == _numel(trainable)
    assert summary["frozen_tensors"] == _n_tensors(frozen)
    assert summary["frozen_params"] == _numel(frozen)

    (group,) = summary["optimizers"][0]["groups"]
    assert group["params_pct_of_model"] == 100.0

    owner_names = {o["name"] for o in group["owners"]}
    assert owner_names == {"trainable"}
    (owner,) = group["owners"]
    # Owner denominator excludes frozen counts (it's per-owner trainable).
    assert owner["n_tensors_of_owner"] == _n_tensors(trainable)
    assert owner["n_params_of_owner"] == _numel(trainable)
    assert owner["tensors_pct_of_owner"] == 100.0
    assert owner["params_pct_of_owner"] == 100.0


def test_unclaimed_trainable_params_produce_sub_100_group_pct():
    claimed = _tiny_module(4, 8)
    unclaimed = _tiny_module(8, 2)
    modules = {"claimed": claimed, "unclaimed": unclaimed}

    opt = SGD(list(claimed.parameters()), lr=0.01)
    summary = build_optimizer_summary([opt], [ConstantLR(opt)], modules)

    (group,) = summary["optimizers"][0]["groups"]
    expected_pct = (
        _numel(claimed) / (_numel(claimed) + _numel(unclaimed)) * 100
    )
    assert abs(group["params_pct_of_model"] - expected_pct) < 1e-9
    assert group["params_pct_of_model"] < 100.0

    # The claimed owner is fully in this group → still 100% of itself.
    (owner,) = group["owners"]
    assert owner["name"] == "claimed"
    assert owner["params_pct_of_owner"] == 100.0
    # Unclaimed owner appears nowhere.
    tensor_totals, _ = _owner_pct_totals(summary)
    assert "unclaimed" not in tensor_totals


def test_multi_group_within_one_optimizer():
    backbone = _tiny_module(4, 8)
    head = _tiny_module(8, 2)
    modules = {"backbone": backbone, "head": head}

    opt = SGD(
        [
            {"params": list(backbone.parameters()), "lr": 0.01},
            {"params": list(head.parameters()), "lr": 0.001},
        ],
        momentum=0.9,
    )
    summary = build_optimizer_summary(
        [opt], [LambdaLR(opt, lr_lambda=lambda _e: 1.0)], modules
    )

    (opt_info,) = summary["optimizers"]
    g0, g1 = opt_info["groups"]
    assert g0["hyperparams"]["lr"] == 0.01
    assert g1["hyperparams"]["lr"] == 0.001

    # Owner-relative pct: each group has one owner fully contained in it.
    for group in (g0, g1):
        (owner,) = group["owners"]
        assert owner["params_pct_of_owner"] == 100.0

    # Group-relative pct: split adds to 100% of model.
    assert (
        abs(g0["params_pct_of_model"] + g1["params_pct_of_model"] - 100.0)
        < 1e-9
    )


def test_hyperparams_filter_out_callables_and_collections():
    m = _tiny_module(2, 2)
    opt = SGD(list(m.parameters()), lr=0.01, momentum=0.9)
    opt.param_groups[0]["fake_callable"] = lambda x: x
    opt.param_groups[0]["fake_list"] = [1, 2, 3]

    summary = build_optimizer_summary([opt], [ConstantLR(opt)], {"m": m})
    hp = summary["optimizers"][0]["groups"][0]["hyperparams"]
    assert "lr" in hp
    assert "momentum" in hp
    assert "fake_callable" not in hp
    assert "fake_list" not in hp
    assert "params" not in hp


def test_external_params_get_own_totals_summing_to_100_percent():
    known = _tiny_module(4, 4)
    orphan = _tiny_module(4, 4)  # NOT in the modules map

    opt = SGD([*known.parameters(), *orphan.parameters()], lr=0.01)
    summary = build_optimizer_summary(
        [opt], [ConstantLR(opt)], {"known": known}
    )

    (group,) = summary["optimizers"][0]["groups"]
    owners_by_name = {o["name"]: o for o in group["owners"]}
    assert "known" in owners_by_name
    assert "<external>" in owners_by_name

    # <external> gets its own denominator = the aggregate of external
    # params across all groups, so its pct still sums to 100% per owner.
    ext = owners_by_name["<external>"]
    assert ext["n_tensors_of_owner"] == _n_tensors(orphan)
    assert ext["n_params_of_owner"] == _numel(orphan)
    assert ext["params_pct_of_owner"] == 100.0


def test_empty_model_produces_safe_zero_totals():
    opt = SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
    summary = build_optimizer_summary([opt], [ConstantLR(opt)], {})

    # All owned params are external. Trainable-model totals equal the
    # external ones (since there's no other source of trainable params).
    assert summary["trainable_tensors"] == 1
    assert summary["trainable_params"] == 1
    (group,) = summary["optimizers"][0]["groups"]
    assert group["params_pct_of_model"] == 100.0
    (owner,) = group["owners"]
    assert owner["name"] == "<external>"
    assert owner["params_pct_of_owner"] == 100.0


def test_plain_optimizer_summary_renders_all_optimizers():
    """`log_optimizer_summary(use_rich=False)` is the fallback used when
    `rich_logging` is disabled.

    It must render every optimizer, group, hyperparameter and owner as
    indented plaintext, since that is the only record of how the
    parameters were split when rich output is unavailable.
    """
    backbone = _tiny_module(4, 8)
    head = _tiny_module(8, 2)
    backbone_optimizer = SGD(backbone.parameters(), lr=0.01)
    head_optimizer = SGD(head.parameters(), lr=0.5)
    summary = build_optimizer_summary(
        [backbone_optimizer, head_optimizer],
        [
            ConstantLR(backbone_optimizer, factor=1.0),
            ConstantLR(head_optimizer, factor=1.0),
        ],
        {"backbone": backbone, "head": head},
    )

    messages: list[str] = []
    handler_id = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="INFO",
    )
    try:
        log_optimizer_summary(summary, use_rich=False)
    finally:
        logger.remove(handler_id)

    rendered = messages[0]

    assert "Using 2 optimizer(s)." in rendered
    assert "trainable: 4 tensors / 58 params" in rendered
    assert "frozen:    0 tensors / 0 params" in rendered
    assert "Optimizer #0: SGD + ConstantLR  (1 parameter group(s))" in rendered
    assert "Optimizer #1: SGD + ConstantLR  (1 parameter group(s))" in rendered
    assert (
        "Group #0: 2 tensors (50.0% of trainable)  •  "
        "40 params (69.0% of trainable)" in rendered
    )
    assert (
        "Group #0: 2 tensors (50.0% of trainable)  •  "
        "18 params (31.0% of trainable)" in rendered
    )
    assert "      lr = 0.01\n" in rendered
    assert "      lr = 0.5\n" in rendered
    assert "backbone\n        tensors 2/2 (100.0% of owner)" in rendered
    assert "head\n        tensors 2/2 (100.0% of owner)" in rendered


def test_rich_optimizer_summary_panels_fit_their_content():
    """The rich panels must shrink to their content instead of
    stretching across the whole terminal.

    Anything laid out with `rich.columns.Columns` measures as wide as
    the console, which silently defeats the enclosing `Panel.fit` and
    blows every panel up to the full width.
    """
    backbone = _tiny_module(4, 8)
    optimizer = SGD(backbone.parameters(), lr=0.01)
    summary = build_optimizer_summary(
        [optimizer],
        [ConstantLR(optimizer, factor=1.0)],
        {"backbone": backbone},
    )

    def render(width: int) -> list[str]:
        buffer = io.StringIO()
        console = Console(
            width=width, force_terminal=False, no_color=True, file=buffer
        )
        with patch("rich.get_console", return_value=console):
            log_optimizer_summary(summary, use_rich=True)
        return [line.rstrip() for line in buffer.getvalue().splitlines()]

    narrow = render(120)
    wide = render(240)

    assert any("Optimizer #0" in line for line in narrow)
    assert any("Group #0" in line for line in narrow)
    # Content-driven layout — doubling the terminal must not widen anything.
    assert narrow == wide
    assert max(len(line) for line in narrow) < 120
