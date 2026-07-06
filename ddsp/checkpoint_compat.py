"""Checkpoint-loading compatibility for PyTorch >= 2.6.

torch 2.6 changed ``torch.load``'s default to ``weights_only=True``. Its safe unpickler then refuses
the OmegaConf config objects that Lightning stores in a checkpoint's ``hyper_parameters`` (Hydra
``DictConfig``/``ListConfig`` and their metadata/value nodes), which breaks *resume* and *export*
of our checkpoints on Colab while still working on older local torch.

``allow_omegaconf_checkpoints()`` uses the sanctioned ``torch.serialization.add_safe_globals`` API
(the fix the ``torch.load`` error message itself recommends) to allowlist exactly those benign
container/metadata/value-node classes, so ``weights_only=True`` loading succeeds. It is a no-op on
torch versions without the API (which default to ``weights_only=False`` anyway). Call it once, before
any checkpoint load — the CLI entry points do this at import time.
"""

import collections
import typing

import torch


def allow_omegaconf_checkpoints() -> None:
  add = getattr(getattr(torch, "serialization", None), "add_safe_globals", None)
  if add is None:  # torch < 2.4: no allowlist API, and weights_only defaults to False anyway.
    return

  safe = [collections.defaultdict, typing.Any]
  try:
    from omegaconf import DictConfig, ListConfig
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf import nodes as _nodes

    safe += [DictConfig, ListConfig, ContainerMetadata, Metadata]
    # Every OmegaConf value-node type — robust to whatever primitive values a config contains.
    for _name in (
      "Node", "ValueNode", "AnyNode", "IntegerNode", "FloatNode", "StringNode",
      "BooleanNode", "BytesNode", "EnumNode", "PathNode", "InterpolationResultNode",
    ):
      _cls = getattr(_nodes, _name, None)
      if _cls is not None:
        safe.append(_cls)
  except Exception:
    pass  # OmegaConf missing/rearranged — register whatever we could resolve.

  # Register per-item so one unregisterable entry can't drop the rest.
  for obj in safe:
    try:
      add([obj])
    except Exception:
      pass
