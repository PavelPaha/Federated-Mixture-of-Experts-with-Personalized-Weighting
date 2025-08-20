#!/usr/bin/env python3
"""
Ensure that all datasets mentioned in data.py have the same set of experiment
configs as wikitext2 (alphas: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0), using the flat
Hydra global format identical to configs under experiments/wikitext2/.

This script creates any missing files; it does not overwrite existing ones.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List


DATASET_NAME_MAP: Dict[str, str] = {
    "wikitext2": "wikitext-2-v1",
    "wikitext103": "wikitext-103-raw-v1",
    "c4": "c4",
    "ptb": "ptb",
    "openwebtext": "openwebtext",
    "lm1b": "lm1b",
}

ALPHAS: List[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def alpha_to_name(a: float) -> str:
    s = f"{a:.1f}"
    whole, frac = s.split(".")
    return f"alpha_{whole}_{frac}"


def build_content(dataset_key: str, alpha_value: float, exp_name: str) -> str:
    lines = []
    lines.append("# @package _global_")
    lines.append("")
    lines.append("defaults:")
    lines.append("  - /base")
    lines.append("  - _self_")
    lines.append("")
    lines.append("experiment:")
    lines.append(f"  name: {exp_name}")
    lines.append("")
    lines.append("training:")
    lines.append(f"  dataset_name: {DATASET_NAME_MAP.get(dataset_key, dataset_key)}")
    lines.append("")
    lines.append("alpha_schedule:")
    lines.append("  _target_: schedulers.ConstantScheduler")
    lines.append("  use_warmup: true")
    lines.append("  warmup_steps: null")
    lines.append(f"  initial_value: {alpha_value}")
    lines.append(f"  final_value: {alpha_value}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    exps_dir = root / "configs" / "experiments"
    datasets = ["wikitext2", "wikitext103", "c4", "ptb", "openwebtext", "lm1b"]
    created = 0

    for ds in datasets:
        ds_dir = exps_dir / ds
        ds_dir.mkdir(parents=True, exist_ok=True)
        for a in ALPHAS:
            alpha_name = alpha_to_name(a)
            target = ds_dir / f"{alpha_name}.yaml"
            if target.exists():
                continue
            exp_name = f"moe_{ds}_{alpha_name}"
            content = build_content(ds, a, exp_name)
            target.write_text(content, encoding="utf-8")
            created += 1
            print(f"Created: {target.relative_to(root)}")

    print(f"Done. Created {created} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())














