#!/usr/bin/env python3
"""One home for every RL-versus-BC endpoint comparison artifact.

Both endpoint programs used to write wherever `--output-dir` happened to point,
so five collections of one experiment ended up in three unrelated directories.
This module gives them a single root:

    program/shiyan/rl_bc_endpoint_comparison/
        formal_eval/                 one directory per endpoint evaluation
        traces_fixed32/              one directory per fixed32 trajectory dump
        traces_s2_pilot/             one directory per S2 pilot dump

The root sits outside the code repository on purpose: the artifacts outlive a
re-clone of the code. It is resolved from this file's location, so a checkout
that moves still finds it, and `ENDPOINT_OUTPUT_ROOT` overrides it for a machine
that has no `program/` directory (a training host, for example).

A run never writes into an existing directory. Both writers refuse to overwrite,
so a fresh dated directory is the only safe target -- and a name collision fails
loudly instead of merging two runs into one.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path(
    os.environ.get(
        "ENDPOINT_OUTPUT_ROOT",
        _REPO_ROOT.parents[1] / "program" / "shiyan" / "rl_bc_endpoint_comparison",
    )
).resolve()

_SLUG = re.compile(r"[^A-Za-z0-9._-]+")


def slug(text):
    """A directory-name-safe form of `text`; never empty."""
    cleaned = _SLUG.sub("_", str(text)).strip("_")
    return cleaned or "run"


def resolve_output_dir(explicit, category, label):
    """The output directory for one run.

    An explicit `--output-dir` always wins, so a caller that needs a specific
    path (a remote collection, a rerun beside an old one) keeps working. With
    no explicit path the run lands at
    `<root>/<category>/<YYYYMMDD_HHMMSS>_<label>`, which identifies the run by
    when it ran and what it evaluated.
    """
    if explicit is not None:
        return Path(explicit).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (DEFAULT_ROOT / category / "{}_{}".format(stamp, slug(label))).resolve()


def outside_default_root(path):
    """True when `path` is not under `DEFAULT_ROOT`.

    The callers print a warning rather than raising: a collection on a machine
    without `program/` is legitimate, but it must not pass unnoticed.
    """
    try:
        Path(path).resolve().relative_to(DEFAULT_ROOT)
    except ValueError:
        return True
    return False
