"""Batch-train several models from a directory of experiment configs.

For every config in the given directory this runs the full pipeline in order:

    synth  ->  compressor + prior  ->  post-net  ->  export (.ts)

Each stage is OPTIONAL and driven by the config itself:
  * synth   : cli.train          — always runs (the base model) unless disabled/skipped
  * prior   : cli.train_prior     — runs when `prior.enabled` is true; it also trains the
                                    LatentCompressor first if one isn't present (so "compressor"
                                    is part of the prior stage, gated by `compressor.force_restart`
                                    + whether a checkpoint already exists)
  * postnet : cli.train_postnet   — runs when `postnet.enabled` is true
  * export  : cli.export          — writes models/<configs-folder>/<experiment.name>.ts (runs by
                                    default; disable per config with `train.export: false`, globally
                                    by omitting it from --stages, pass export flags via --export-args)

The three underlying CLIs already self-gate on their `enabled` flags, so a disabled stage is a
fast no-op. On top of that you can steer the batch with:
  * an optional `train:` block per config: {synth: bool, prior: bool, postnet: bool}
  * --stages to restrict globally, --skip-existing to skip stages whose checkpoint already exists,
    --configs/--exclude to pick configs, and --dry-run to preview the plan.

Configs are loaded via Hydra's `--config-dir <dir> -cn <stem>`, so the directory can live anywhere
(the configs should be self-contained — the same flat form as configs/rafael.yaml).

Examples:
  python -m cli.train_multiple configs/batch
  python -m cli.train_multiple experiments/ --stages synth,prior --skip-existing
  python -m cli.train_multiple configs/batch --dry-run
  python -m cli.train_multiple configs/batch --extra prior.training.force_restart=true
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import yaml


STAGE_ORDER = ["synth", "prior", "postnet", "export"]
STAGE_MODULE = {
  "synth": "cli.train",
  "prior": "cli.train_prior",
  "postnet": "cli.train_postnet",
  "export": "cli.export",  # invoked differently (argparse, not Hydra) — see _run_stage
}


def _load_yaml(path: str) -> Dict[str, Any]:
  """Best-effort load of an experiment config for gating decisions.

  Only used to read literal flags (experiment.name, *.enabled). Never fatal: on any parse error we
  return {} and let the underlying CLI self-gate.
  """
  try:
    with open(path, "r") as f:
      data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}
  except Exception as e:  # noqa: BLE001 - gating must never crash the batch
    print(f"  [warn] could not parse {path}: {e}; assuming stages run and letting the CLI self-gate")
    return {}


def _get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
  cur: Any = d
  for k in keys:
    if not isinstance(cur, dict) or k not in cur:
      return default
    cur = cur[k]
  return cur


def _discover_configs(configs_dir: str, only: Optional[List[str]], exclude: List[str]) -> List[str]:
  """Return sorted config file paths (*.yaml/*.yml), skipping `_`-prefixed partials."""
  names = []
  for fn in sorted(os.listdir(configs_dir)):
    if not (fn.endswith(".yaml") or fn.endswith(".yml")):
      continue
    if fn.startswith("_"):
      continue
    stem = os.path.splitext(fn)[0]
    if only is not None and stem not in only:
      continue
    if stem in exclude:
      continue
    names.append(os.path.join(configs_dir, fn))
  return names


def _export_ts_path(cfg: Dict[str, Any], stem: str, models_subdir: str) -> str:
  """Where this config's exported .ts goes: <models_subdir>/<experiment.name>.ts."""
  name = _get(cfg, "experiment", "name", default=stem)
  return os.path.join(models_subdir, f"{name}.ts")


def _find_ckpt(directory: str, typ: str = "last") -> Optional[str]:
  """Recursively find a `*<typ>*.ckpt` under directory (newest by ctime), else None."""
  if not os.path.isdir(directory):
    return None
  hits = []
  for root, _dirs, files in os.walk(directory):
    for f in files:
      if typ in f and f.endswith(".ckpt"):
        hits.append(os.path.join(root, f))
  if not hits:
    return None
  return max(hits, key=os.path.getctime)


def _stage_checkpoint_exists(stage: str, cfg: Dict[str, Any], models_subdir: str = "models") -> bool:
  """Whether the primary checkpoint a stage would produce already exists (for --skip-existing)."""
  name = _get(cfg, "experiment", "name", default=None)
  training_dir = _get(cfg, "experiment", "training_dir", default="training")
  if not name:
    return False
  if stage == "synth":
    d = os.path.join(training_dir, "synth", name)
    return bool(_find_ckpt(d, "last") or _find_ckpt(d, "best"))
  if stage == "prior":
    discrete = bool(_get(cfg, "prior", "discrete", "enabled", default=False))
    sub = "prior_discrete" if discrete else "prior"
    d = os.path.join(training_dir, sub, name)
    return bool(_find_ckpt(d, "last") or _find_ckpt(d, "best"))
  if stage == "postnet":
    d = os.path.join(training_dir, "postnet", name)
    return bool(_find_ckpt(d, "best") or _find_ckpt(d, "last"))
  if stage == "export":
    return os.path.isfile(os.path.join(models_subdir, f"{name}.ts"))
  return False


def _stage_enabled_in_config(stage: str, cfg: Dict[str, Any]) -> bool:
  """Default per-config gating (mirrors what the underlying CLI would decide), before any --stages
  restriction or --skip-existing. An optional top-level `train:` block overrides these defaults."""
  train_block = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
  if stage in train_block:
    return bool(train_block[stage])
  if stage == "synth":
    return True  # the base model; run unless explicitly turned off
  if stage == "prior":
    return bool(_get(cfg, "prior", "enabled", default=False))
  if stage == "postnet":
    return bool(_get(cfg, "postnet", "enabled", default=False))
  if stage == "export":
    return True  # produce the deployable .ts by default; disable via `train.export: false` or --stages
  return False


def _plan_for_config(path: str, requested_stages: List[str], skip_existing: bool, models_subdir: str
                     ) -> Tuple[str, str, Dict[str, Any], List[Tuple[str, str]]]:
  """Return (path, stem, cfg, [(stage, decision)]) where decision in {run, skip:<reason>}."""
  stem = os.path.splitext(os.path.basename(path))[0]
  cfg = _load_yaml(path)
  decisions: List[Tuple[str, str]] = []
  for stage in STAGE_ORDER:
    if stage not in requested_stages:
      continue
    if not _stage_enabled_in_config(stage, cfg):
      decisions.append((stage, "skip:disabled-in-config"))
      continue
    if skip_existing and _stage_checkpoint_exists(stage, cfg, models_subdir):
      decisions.append((stage, "skip:checkpoint-exists"))
      continue
    decisions.append((stage, "run"))
  return path, stem, cfg, decisions


def _has_arg(args: List[str], name: str) -> bool:
  return any(a == name or a.startswith(name + "=") for a in args)


def _stage_cmd(stage: str, stem: str, config_path: str, configs_dir_abs: str, extra: List[str],
               export_args: List[str], cfg: Dict[str, Any], models_subdir: str) -> List[str]:
  """Build the subprocess argv for a stage.

  Training stages are Hydra apps (`--config-dir <dir> -cn <stem>` + `extra` overrides). Export is a
  plain argparse CLI that takes the config FILE path; we point its output at
  <models_subdir>/<name>.ts (unless the user set --output_path themselves via --export-args).
  """
  if stage == "export":
    args = list(export_args)
    if not _has_arg(args, "--output_path"):
      args += ["--output_path", _export_ts_path(cfg, stem, models_subdir)]
    return [sys.executable, "-m", STAGE_MODULE[stage], "--config", config_path] + args
  return [sys.executable, "-m", STAGE_MODULE[stage],
          "--config-dir", configs_dir_abs, "-cn", stem] + extra


def _run_stage(cmd: List[str], log_path: str, env: Dict[str, str]) -> int:
  """Run a stage subprocess, streaming its output to a log file. Returns rc."""
  os.makedirs(os.path.dirname(log_path), exist_ok=True)
  with open(log_path, "w") as logf:
    logf.write("$ " + " ".join(cmd) + "\n\n")
    logf.flush()
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    try:
      proc.wait()
    except KeyboardInterrupt:
      proc.terminate()
      try:
        proc.wait(timeout=20)
      except subprocess.TimeoutExpired:
        proc.kill()
      raise
  return int(proc.returncode)


def _tail(path: str, n: int = 15) -> str:
  try:
    with open(path, "r") as f:
      lines = f.readlines()
    return "".join(lines[-n:])
  except Exception:
    return "(no log)"


# ── metric collection (for the train_report.txt written into the configs dir) ────────────────
# The "accuracy"/quality scalar each stage logs to TensorBoard, and how to reduce it over the run.
_METRIC_SPECS = {
  "synth":      [("val_loss", "min"), ("val/MultiResolutionSTFTLoss", "min")],
  "compressor": [("val_loss", "min"), ("val_perplexity", "max")],
  "prior":      [("val_acc", "max"), ("val_loss", "min"), ("val_style_sep_acc", "max")],
  "postnet":    [("val_loss", "min"), ("val_mrstft", "min")],
}
_METRIC_PRETTY = {
  "val/MultiResolutionSTFTLoss": "val_mrstft",
  "val_style_sep_acc": "val_style_sep",
}
_REPORT_STAGES = ["synth", "compressor", "prior", "postnet", "export"]  # compressor = part of prior


def _metric_dir(stage: str, cfg: Dict[str, Any]) -> Optional[str]:
  name = _get(cfg, "experiment", "name", default=None)
  training_dir = _get(cfg, "experiment", "training_dir", default="training")
  if not name:
    return None
  if stage == "synth":
    return os.path.join(training_dir, "synth", name)
  if stage == "compressor":
    return os.path.join(training_dir, "compressor", name)
  if stage == "prior":
    sub = "prior_discrete" if bool(_get(cfg, "prior", "discrete", "enabled", default=False)) else "prior"
    return os.path.join(training_dir, sub, name)
  if stage == "postnet":
    return os.path.join(training_dir, "postnet", name)
  return None


def _read_best_scalars(base_dir: Optional[str], specs: List[Tuple[str, str]]) -> Dict[str, float]:
  """Best scalar per spec, reduced across ALL TensorBoard event files under base_dir.

  We aggregate over every event file (not just the newest) because Lightning starts a fresh
  `version_N` on each run/resume, and a no-op resume (training already complete) writes an EMPTY
  event file — so the newest file often has none of the metric tags. Taking the best over all
  versions is robust to that and to multi-session (resumed) training.
  """
  if not base_dir or not os.path.isdir(base_dir):
    return {}
  import glob
  evs = glob.glob(os.path.join(base_dir, "**", "events.out.tfevents*"), recursive=True)
  if not evs:
    return {}
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  out: Dict[str, float] = {}
  for ev in evs:
    try:
      ea = EventAccumulator(ev, size_guidance={"scalars": 0})
      ea.Reload()
      have = set(ea.Tags().get("scalars", []))
    except Exception:  # noqa: BLE001 - a corrupt/partial event file must not abort reporting
      continue
    for tag, mode in specs:
      if tag not in have:
        continue
      vals = [s.value for s in ea.Scalars(tag)]
      if not vals:
        continue
      v = max(vals) if mode == "max" else min(vals)
      if tag not in out:
        out[tag] = v
      else:
        out[tag] = max(out[tag], v) if mode == "max" else min(out[tag], v)
  return out


def _fmt_metrics(metrics: Dict[str, float]) -> str:
  if not metrics:
    return "no metrics found"
  parts = []
  for tag, val in metrics.items():
    name = _METRIC_PRETTY.get(tag, tag)
    parts.append(f"{name}={val:.4f}" if abs(val) < 1e4 else f"{name}={val:.4g}")
  return "  ".join(parts)


def _write_report(report_path: str, ts: str, configs_dir_abs: str, extra: List[str],
                  plans: List[Tuple[str, str, Dict[str, Any], List[Tuple[str, str]]]],
                  status_map: Dict[Tuple[str, str], str], models_subdir: str) -> None:
  lines = [
    "=" * 78,
    f"Training report — {ts}",
    f"configs dir : {configs_dir_abs}",
  ]
  if extra:
    lines.append(f"extra hydra : {' '.join(extra)}")
  lines.append("=" * 78)
  lines.append("")
  for _path, stem, cfg, _decisions in plans:
    lines.append(f"[{stem}]")
    for stage in _REPORT_STAGES:
      if stage == "compressor":
        status = "(within prior stage)"
      else:
        status = status_map.get((stem, stage), "(not requested)")
      if stage == "export":
        ts_path = _export_ts_path(cfg, stem, models_subdir)
        if os.path.isfile(ts_path):
          mb = os.path.getsize(ts_path) / 1e6
          detail = f"{ts_path} ({mb:.0f} MB)"
        else:
          detail = "no .ts found"
      else:
        detail = _fmt_metrics(_read_best_scalars(_metric_dir(stage, cfg), _METRIC_SPECS[stage]))
      lines.append(f"  {stage:11s} {status:26s} {detail}")
    lines.append("")
  lines.append("Notes: prior 'val_acc' is teacher-forced next-token accuracy on held-out windows;")
  lines.append("losses are best (min) over the run, perplexity/accuracy are best (max).")
  with open(report_path, "w") as f:
    f.write("\n".join(lines) + "\n")


def main() -> int:
  ap = argparse.ArgumentParser(
    description="Batch-train (and export) synth/compressor/prior/postnet for every config in a dir.")
  ap.add_argument("configs_dir", help="Directory containing experiment configs (*.yaml).")
  ap.add_argument("--stages", default=",".join(STAGE_ORDER),
                  help=f"Comma list of stages to run, from {STAGE_ORDER} (default: all). "
                       "compressor is part of the 'prior' stage.")
  ap.add_argument("--configs", default=None,
                  help="Comma list of config stems to include (default: all in the directory).")
  ap.add_argument("--exclude", default="",
                  help="Comma list of config stems to skip.")
  ap.add_argument("--skip-existing", action="store_true",
                  help="Skip a stage whose output checkpoint already exists.")
  ap.add_argument("--stop-on-error", action="store_true",
                  help="Abort the whole batch on the first stage failure (default: continue).")
  ap.add_argument("--dry-run", action="store_true",
                  help="Print the plan (configs, stages, commands) without training.")
  ap.add_argument("--export-args", default="",
                  help="Extra flags passed to the export stage only (a single string, shlex-split), "
                       "e.g. --export-args \"--no_postnet --no_emit_terrain\".")
  ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                  help="Everything after --extra is passed verbatim as Hydra overrides to every "
                       "TRAINING stage (e.g. --extra prior.training.force_restart=true). Put it last. "
                       "Does not apply to export — use --export-args for that.")
  ap.add_argument("--log-dir", default=None,
                  help="Where to write per-stage logs (default: ./train_multiple_logs/<timestamp>).")
  ap.add_argument("--report-name", default="train_report.txt",
                  help="Filename of the accuracy record written into the configs dir "
                       "(default: train_report.txt). Set empty or use --no-report to disable.")
  ap.add_argument("--no-report", action="store_true",
                  help="Do not write the train_report.txt accuracy record.")
  args = ap.parse_args()

  configs_dir = args.configs_dir
  if not os.path.isdir(configs_dir):
    print(f"error: not a directory: {configs_dir}", file=sys.stderr)
    return 2
  configs_dir_abs = os.path.abspath(configs_dir)

  requested_stages = [s.strip() for s in args.stages.split(",") if s.strip()]
  bad = [s for s in requested_stages if s not in STAGE_ORDER]
  if bad:
    print(f"error: unknown stage(s) {bad}; valid: {STAGE_ORDER}", file=sys.stderr)
    return 2
  requested_stages = [s for s in STAGE_ORDER if s in requested_stages]  # canonical order

  only = [s.strip() for s in args.configs.split(",")] if args.configs else None
  exclude = [s.strip() for s in args.exclude.split(",") if s.strip()]
  extra = list(args.extra or [])
  import shlex
  export_args = shlex.split(args.export_args or "")

  config_paths = _discover_configs(configs_dir, only, exclude)
  if not config_paths:
    print(f"No configs found in {configs_dir} (after include/exclude filters).")
    return 1

  ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
  log_dir = args.log_dir or os.path.join("train_multiple_logs", ts)

  # Exports land in models/<configs-folder-basename>/<name>.ts, grouping a config set's outputs.
  models_subdir = os.path.join("models", os.path.basename(configs_dir_abs.rstrip(os.sep)))

  env = dict(os.environ)
  env.setdefault("PRIOR_NO_PBAR", "1")  # keep prior logs readable in files (no progress bar spam)

  # ── plan ──────────────────────────────────────────────────────────────────────────────────
  plans = [_plan_for_config(p, requested_stages, args.skip_existing, models_subdir) for p in config_paths]
  print(f"train_multiple: {len(plans)} config(s) from {configs_dir_abs}")
  print(f"stages={requested_stages}  skip_existing={args.skip_existing}  "
        f"stop_on_error={args.stop_on_error}")
  if "export" in requested_stages:
    print(f"exports -> {models_subdir}/<name>.ts")
  if extra:
    print(f"extra overrides (training stages): {' '.join(extra)}")
  print(f"logs -> {log_dir}\n")
  for _path, stem, _cfg, decisions in plans:
    parts = [f"{st}={dec}" for st, dec in decisions]
    print(f"  • {stem}: " + ("  ".join(parts) if parts else "(no requested stages)"))
  print()

  # Warn on duplicate experiment.name: those configs write to the SAME training/<stage>/<name>
  # dirs and overwrite each other (and the report can't tell them apart). Almost always a mistake
  # from copying a config without changing `experiment.name`.
  by_name: Dict[str, List[str]] = {}
  for _path, stem, cfg, _d in plans:
    nm = _get(cfg, "experiment", "name", default=None)
    if nm:
      by_name.setdefault(str(nm), []).append(stem)
  dups = {nm: stems for nm, stems in by_name.items() if len(stems) > 1}
  if dups:
    print("  !! WARNING: multiple configs share the same experiment.name — they will train into the")
    print("     SAME training/<stage>/<name> dirs and OVERWRITE each other. Give each a unique name:")
    for nm, stems in dups.items():
      print(f"       name '{nm}': {', '.join(stems)}")
    print()

  if args.dry_run:
    print("dry-run: nothing executed. Commands that WOULD run:")
    for path, stem, cfg, decisions in plans:
      for st, dec in decisions:
        if dec == "run":
          print("  $ " + " ".join(
            _stage_cmd(st, stem, path, configs_dir_abs, extra, export_args, cfg, models_subdir)))
    return 0

  # ── run ───────────────────────────────────────────────────────────────────────────────────
  if "export" in requested_stages:
    os.makedirs(models_subdir, exist_ok=True)  # export's torch.jit.save needs the dir to exist
  results: List[Tuple[str, str, str, float]] = []  # (stem, stage, status, seconds)
  aborted = False
  for ci, (path, stem, cfg, decisions) in enumerate(plans, 1):
    if aborted:
      break
    for st, dec in decisions:
      if dec != "run":
        results.append((stem, st, dec, 0.0))
        continue
      log_path = os.path.join(log_dir, f"{stem}__{st}.log")
      banner = f">>> [{ci}/{len(plans)}] {stem} :: {st}"
      print(f"{banner} :: RUNNING  (log: {log_path})", flush=True)
      t0 = time.time()
      try:
        cmd = _stage_cmd(st, stem, path, configs_dir_abs, extra, export_args, cfg, models_subdir)
        rc = _run_stage(cmd, log_path, env)
      except KeyboardInterrupt:
        dt = time.time() - t0
        print(f"<<< {stem} :: {st} :: INTERRUPTED after {dt:.0f}s", flush=True)
        results.append((stem, st, "interrupted", dt))
        aborted = True
        break
      dt = time.time() - t0
      if rc == 0:
        print(f"<<< {stem} :: {st} :: OK ({dt:.0f}s)", flush=True)
        results.append((stem, st, "ok", dt))
      else:
        print(f"<<< {stem} :: {st} :: FAILED (rc={rc}, {dt:.0f}s)", flush=True)
        print("    last lines of log:")
        for line in _tail(log_path).splitlines():
          print("    | " + line)
        results.append((stem, st, f"failed(rc={rc})", dt))
        # a failed stage means downstream stages of THIS config can't be trusted -> skip them
        remaining = [s for s, d in decisions if d == "run" and STAGE_ORDER.index(s) > STAGE_ORDER.index(st)]
        for s in remaining:
          results.append((stem, s, "skipped:upstream-failed", 0.0))
        if args.stop_on_error:
          aborted = True
        break  # move to next config (or abort)

  # ── summary ───────────────────────────────────────────────────────────────────────────────
  print("\n================= summary =================")
  n_ok = n_fail = 0
  status_map: Dict[Tuple[str, str], str] = {}
  for stem, st, status, dt in results:
    status_map[(stem, st)] = status
    mark = "ok " if status == "ok" else ("FAIL" if status.startswith("failed") else "-- ")
    if status == "ok":
      n_ok += 1
    elif status.startswith("failed"):
      n_fail += 1
    tstr = f"{dt:6.0f}s" if dt else "      "
    print(f"  [{mark}] {stem:24s} {st:8s} {tstr}  {status}")
  print(f"\n{n_ok} stage(s) ok, {n_fail} failed" + ("  (ABORTED)" if aborted else ""))
  print(f"logs: {log_dir}")

  # ── accuracy record written into the configs dir ────────────────────────────────────────────
  if not args.no_report and args.report_name:
    report_path = os.path.join(configs_dir_abs, args.report_name)
    try:
      _write_report(report_path, ts, configs_dir_abs, extra, plans, status_map, models_subdir)
      print(f"report: {report_path}")
    except Exception as e:  # noqa: BLE001
      print(f"[warn] could not write report to {report_path}: {e}")

  return 1 if (n_fail or aborted) else 0


if __name__ == "__main__":
  raise SystemExit(main())
