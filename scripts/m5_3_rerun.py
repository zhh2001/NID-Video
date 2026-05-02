"""Backward-compatible alias for scripts/baseline_rerun.py.

The earlier M5.3 work referenced this script by name (the commit body and
the README inside ``outputs/run_20260501_162117/m5_3_rerun/`` both quote
``scripts/m5_3_rerun.py`` as the reproduction command). The implementation
has since been generalised into ``scripts/baseline_rerun.py``; this file
forwards to it with the M5.3 defaults (``--task-label`` /
``--script-name``) injected when the caller does not supply them, so
existing reproduction commands keep working unchanged.
"""

from __future__ import annotations

import sys

from scripts.baseline_rerun import main


_M5_3_DEFAULTS = {
    "--task-label": "M5.3 noise-free baseline (M5.2 best.pt)",
    "--script-name": "scripts/m5_3_rerun.py",
}


def _inject_defaults(argv: list[str]) -> list[str]:
    """Insert each M5.3 default that the caller has not already overridden."""
    out = list(argv)
    for flag, value in _M5_3_DEFAULTS.items():
        if not any(a == flag or a.startswith(flag + "=") for a in out):
            out += [flag, value]
    return out


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *_inject_defaults(sys.argv[1:])]
    raise SystemExit(main())
