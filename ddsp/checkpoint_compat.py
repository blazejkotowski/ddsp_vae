"""Checkpoint-loading compatibility for PyTorch >= 2.6.

torch 2.6 changed ``torch.load``'s default to ``weights_only=True``, whose safe unpickler refuses the
plain Python objects Lightning stores in a checkpoint's ``hyper_parameters`` (Hydra ``DictConfig``,
nested ``dict``/``defaultdict``, our own ``ControlSpace`` dataclasses, …). This breaks *resume* and
*export* of our own — trusted, locally produced — checkpoints on Colab, while older local torch
(default ``weights_only=False``) is unaffected.

The robust fix is to load these trusted checkpoints with ``weights_only=False`` rather than trying to
allowlist every class they contain. Recent Lightning threads a ``weights_only`` argument through
``Trainer.fit`` / ``load_from_checkpoint`` / the checkpoint IO, so we pass ``weights_only=False``
there — guarded by :func:`weights_only_false_kwargs` so it is simply omitted on older Lightning that
lacks the parameter (where torch also still defaults to ``weights_only=False``, so nothing is needed).
For our direct ``torch.load`` calls we pass ``weights_only=False`` explicitly.

:func:`allow_full_checkpoints` is a best-effort fallback (the sanctioned ``add_safe_globals`` API) for
any load path that does not expose a ``weights_only`` argument.
"""

import collections
import inspect
import typing

import torch


def weights_only_false_kwargs(fn) -> dict:
  """``{'weights_only': False}`` if ``fn`` accepts that parameter, else ``{}``.

  Lets us force full (trusted) unpickling on torch>=2.6 via Lightning's ``fit`` /
  ``load_from_checkpoint`` without breaking older Lightning whose signatures lack the parameter."""
  try:
    params = inspect.signature(fn).parameters
  except (TypeError, ValueError):
    return {}
  return {"weights_only": False} if "weights_only" in params else {}


def allow_full_checkpoints() -> None:
  """Best-effort allowlist (torch's ``add_safe_globals``) for load paths without a ``weights_only``
  arg. No-op on torch<2.4 (which defaults ``weights_only=False`` anyway)."""
  add = getattr(getattr(torch, "serialization", None), "add_safe_globals", None)
  if add is None:
    return

  safe = [
    dict, list, tuple, set, frozenset, bytes, bytearray, complex,
    collections.OrderedDict, collections.defaultdict, typing.Any,
  ]
  try:
    from omegaconf import DictConfig, ListConfig
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf import nodes as _nodes
    safe += [DictConfig, ListConfig, ContainerMetadata, Metadata]
    for _name in (
      "Node", "ValueNode", "AnyNode", "IntegerNode", "FloatNode", "StringNode",
      "BooleanNode", "BytesNode", "EnumNode", "PathNode", "InterpolationResultNode",
    ):
      _cls = getattr(_nodes, _name, None)
      if _cls is not None:
        safe.append(_cls)
  except Exception:
    pass
  try:
    from ddsp.interfaces import ControlSpace, ControlField
    safe += [ControlSpace, ControlField]
  except Exception:
    pass

  for obj in safe:  # per-item so one unregisterable entry can't drop the rest
    try:
      add([obj])
    except Exception:
      pass
