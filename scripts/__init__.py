"""Compatibility shim for the original flat ``scripts.`` import namespace.

In the original research repository every analysis script lived in a single flat
``scripts/`` package, so the scripts import one another as
``from scripts.run_xxx import ...``. In this release the same files are grouped
into readable sections under ``analysis/`` (e.g. ``analysis/1_frozen_v2A_audit/``).
Those directory names start with digits and cannot be Python package names, so
this shim makes ``scripts`` a single flat namespace whose modules are found in the
grouped ``analysis/`` folders. It lets the original cross-imports resolve unchanged.

Put the repository root on PYTHONPATH (or run from it) and ``from scripts.X import
Y`` will resolve to ``analysis/<section>/X.py``.
"""
from pathlib import Path

_analysis = Path(__file__).resolve().parent.parent / "analysis"
__path__ = [str(p) for p in sorted(_analysis.iterdir()) if p.is_dir()]
