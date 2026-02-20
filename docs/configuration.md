# OptiRoulette Configuration Guide

This document explains the main settings you can define when creating an
`OptiRoulette` optimizer.

## Quick Start

```python
from optiroulette import OptiRoulette

optimizer = OptiRoulette(model.parameters())
```

With no extra arguments, `OptiRoulette` loads defaults from bundled
`optimized.yaml`:
- optimizer specs
- roulette switching settings
- pool settings (active/backup setup)
- LR scaling rules
- default seed

## Constructor Parameters

Main constructor:

```python
OptiRoulette(
    params,
    *,
    optimizer_specs=None,
    optimizers=None,
    roulette=None,
    pool_config=None,
    lr_scaling_rules=None,
    active_names=None,
    backup_names=None,
    switch_granularity=None,
    switch_every_steps=None,
    switch_probability=None,
    avoid_repeat=None,
    warmup_optimizer=None,
    drop_after_warmup=False,
    warmup_epochs=0,
    warmup_config=None,
    seed=None,
    logger=None,
)
```

### `optimizer_specs`

Defines which optimizers are created by name and with which kwargs.

Supported input formats:

```python
# Dict format (recommended)
optimizer_specs = {
    "adam": {"lr": 1e-3},
    "adamw": {"lr": 8e-4, "weight_decay": 0.01},
}

# List[str] format
optimizer_specs = ["adam", "adamw", "sgd"]

# List[tuple] format
optimizer_specs = [("adam", {"lr": 1e-3}), ("sgd", {"lr": 0.05, "momentum": 0.9})]

# List[dict] format (must include "name")
optimizer_specs = [
    {"name": "adam", "lr": 1e-3},
    {"name": "lion", "lr": 1e-4, "betas": (0.9, 0.99)},
]
```

Notes:
- If you pass `optimizer_specs`, only those optimizers are in the game.
- Names are case-insensitive in resolution, but keep names consistent across
  `optimizer_specs`, `active_names`, and `backup_names`.
- Advanced optimizers (e.g. `adabelief`, `ranger`, `lion`) come from
  `pytorch-optimizer`.

### `optimizers`

Pass pre-built optimizer instances directly:

```python
import torch

optimizers = {
    "adam": torch.optim.Adam(model.parameters(), lr=1e-3),
    "sgd": torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9),
}
optimizer = OptiRoulette(model.parameters(), optimizers=optimizers)
```

If `optimizers` is provided, `optimizer_specs` is ignored.

### `roulette`

Dictionary to set switch and warmup behavior in one place.

Common keys:
- `switch_granularity`: `"epoch"` or `"batch"`
- `switch_every_steps`: integer interval for batch mode
- `switch_probability`: probability to switch at each eligible point
- `avoid_repeat`: avoid selecting current optimizer again
- `warmup_optimizer`: optimizer name used during warmup
- `warmup_epochs`: fixed warmup length
- `drop_after_warmup`: block warmup optimizer after warmup
- `warmup_config`: plateau warmup settings (see below)

Example:

```python
roulette = {
    "switch_granularity": "epoch",
    "avoid_repeat": True,
    "warmup_optimizer": "sgd",
    "warmup_epochs": 12,
    "drop_after_warmup": False,
}
```

### Direct switch/warmup args

You can pass these directly instead of through `roulette`:
- `switch_granularity`
- `switch_every_steps`
- `switch_probability`
- `avoid_repeat`
- `warmup_optimizer`
- `drop_after_warmup`
- `warmup_epochs`
- `warmup_config`

For most fields, precedence is:
1. explicit direct argument
2. `roulette` dict value
3. package default (from bundled config)

Nuance:
- `drop_after_warmup`: direct `True` forces enable. To force disable, set
  `roulette={"drop_after_warmup": False}`.
- `warmup_epochs`: direct positive value overrides. To force `0`, set
  `roulette={"warmup_epochs": 0}`.

### `warmup_config`

Used for plateau-based warmup exit.

Keys:
- `min_val_acc`: require this validation accuracy before ending warmup
- `plateau_threshold`: slope threshold for detecting plateau
- `min_epochs`: minimum epochs before plateau exit is allowed

Example:

```python
warmup_config = {
    "min_val_acc": 0.50,
    "plateau_threshold": 0.001,
    "min_epochs": 10,
}
```

### `pool_config`

Either a `dict` or a `PoolConfig` object. Controls active/backup pools and
failure-based swaps.

Important fields:
- `num_active`
- `num_backup`
- `enable_failure_swaps`
- `failure_threshold`
- `consecutive_failure_limit`
- `catastrophic_drop`
- `catastrophic_failure_limit`
- `blacklist_threshold`
- `swap_recovery` (state transfer and recovery behavior)
- `compatibility_*` fields (advanced compatibility logic)
- `lr_scaling_rules`

The `active_names` list length must match `num_active`. `backup_names` must
contain at least `num_backup` entries.

### `active_names` and `backup_names`

Explicitly choose which optimizers start active and which are backups.

```python
optimizer = OptiRoulette(
    model.parameters(),
    optimizer_specs={
        "adam": {"lr": 1e-3},
        "adamw": {"lr": 8e-4, "weight_decay": 0.01},
        "lion": {"lr": 1e-4, "betas": (0.9, 0.99)},
    },
    pool_config={"num_active": 2, "num_backup": 1},
    active_names=["adam", "adamw"],
    backup_names=["lion"],
)
```

### `lr_scaling_rules`

Optional compatibility scaling rules applied when switching optimizers (for
example, adaptive -> momentum transitions).

If you use full package defaults (`OptiRoulette(model.parameters())`), default
rules from bundled config are applied automatically.

### `seed`

Random seed for optimizer selection RNG.

- `seed=None` uses package default seed from bundled config (`system.seed`,
  fallback `42`).
- You can pass an integer for explicit deterministic behavior.

### `logger`

Custom `logging.Logger` instance used for switch/pool logs.

## Lifecycle Hooks (Recommended for Full Behavior)

`OptiRoulette` is a `torch.optim.Optimizer`, but switching logic is driven by
hooks:

- `on_epoch_start(epoch)`
- `on_batch_start(batch_idx)`
- `on_epoch_end(val_acc=...)`

If you do not call hooks, optimizer stepping still works, but roulette behavior
is limited and warmup/switch transitions will not run as intended.

## Default Injection Rules

This behavior is important:

1. `OptiRoulette(model.parameters())`
   - injects default specs + roulette + pool + LR scaling + seed
2. `OptiRoulette(..., optimizer_specs=...)`
   - uses your specs
   - does not auto-inject default pool/LR scaling unless you pass them
3. `OptiRoulette(..., optimizers=...)`
   - uses your prebuilt optimizer objects
   - does not auto-inject default pool/LR scaling unless you pass them

## Torch-Compatible Kwargs

The constructor currently includes torch-like kwargs such as `lr`, `betas`,
`eps`, `weight_decay`, `amsgrad` for API compatibility. In practice, per-
optimizer values should be defined inside `optimizer_specs` (or inside prebuilt
`optimizers`) because those are what actually configure underlying optimizers.

## Full Example

```python
from optiroulette import OptiRoulette

optimizer = OptiRoulette(
    model.parameters(),
    optimizer_specs={
        "sgd": {"lr": 0.05, "momentum": 0.9, "nesterov": True},
        "adamw": {"lr": 8e-4, "weight_decay": 0.01},
        "lion": {"lr": 1e-4, "betas": (0.9, 0.99)},
    },
    roulette={
        "switch_granularity": "epoch",
        "switch_probability": 1.0,
        "avoid_repeat": True,
        "warmup_optimizer": "sgd",
        "warmup_epochs": 10,
        "drop_after_warmup": True,
    },
    pool_config={
        "num_active": 2,
        "num_backup": 1,
        "enable_failure_swaps": False,
    },
    active_names=["sgd", "adamw"],
    backup_names=["lion"],
    seed=42,
)
```
