"""Paper-figure plotting helper: shared matplotlib style + I/O conventions.

All figure scripts in this project must call ``configure_paper_style()``
before any ``plt`` calls, then use ``load_per_epoch()`` to read retrofit
or forward-instrumented training data, and ``save_figure()`` to emit
the PDF + PNG pair.

Style choices (locked into ``configure_paper_style`` so individual
figure scripts can't drift):

  * Times New Roman serif font everywhere (``font.family = "serif"``,
    ``font.serif = ["Times New Roman"]``); LaTeX-equivalent appearance
    without a TeX dependency.
  * dpi=300 for both display and savefig — vector PDF stays vector,
    PNG is high-resolution for online sharing.
  * ``axes.titlesize = 0`` is a guardrail — any accidental
    ``ax.set_title("...")`` call still emits the text but at zero font
    size, so figure caption text is forced into LaTeX rather than
    embedded in the figure pixels (paper convention).
  * ``savefig.bbox = "tight"`` clips whitespace consistently across
    figures; no per-figure tweaking needed.

The ``load_per_epoch`` helper is a thin reader for the
``<run_dir>/metrics/per_epoch.json`` schema produced by the
``MetricsWriter`` (forward instrumentation) and
``scripts/baseline_rerun.py --ckpt-glob`` (retrofit). Both write
identical schemas; figure code is agnostic to which source produced a
given run's data.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


def configure_paper_style() -> None:
    """Lock matplotlib rcParams to the project's paper-figure conventions.

    Idempotent — re-calling does not stack changes. Must be called
    BEFORE any ``plt.subplots`` / ``ax.plot`` etc. in a figure script;
    rcParams set after a figure is constructed do not propagate.
    """
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman"]
    # Math text falls back to STIX (Times-equivalent) so embedded
    # equations in axis labels render in the same family.
    mpl.rcParams["mathtext.fontset"] = "stix"
    # Title text is rendered at zero size — figure captions belong in
    # the paper LaTeX, not embedded in the figure raster. An accidental
    # ax.set_title("...") call still emits the layout but the text is
    # invisible.
    mpl.rcParams["axes.titlesize"] = 0
    mpl.rcParams["figure.dpi"] = 300
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["savefig.bbox"] = "tight"
    mpl.rcParams["axes.labelsize"] = 11
    mpl.rcParams["xtick.labelsize"] = 10
    mpl.rcParams["ytick.labelsize"] = 10
    mpl.rcParams["legend.fontsize"] = 10


def load_per_epoch(run_dir: Path) -> dict:
    """Read ``<run_dir>/metrics/per_epoch.json`` and return the parsed
    dict. Schema: ``{run_id, config, epochs: [{epoch, grad_steps,
    wall_time_s, metrics: {combined: {...}, fast?: {...}, slow?: {...}}},
    ...]}``. ``fast`` and ``slow`` keys are present in retrofit-produced
    files (every M5-era run after Part 2) and absent in forward-only
    instrumentation; figure code should ``.get(...)`` and skip missing
    splits.
    """
    p = Path(run_dir) / "metrics" / "per_epoch.json"
    if not p.is_file():
        raise FileNotFoundError(
            f"per_epoch.json not found at {p} — has the run been retrofitted?"
        )
    return json.loads(p.read_text())


def save_figure(fig, name: str, output_dir: Path = Path("figures")) -> None:
    """Save ``fig`` as ``<output_dir>/<name>.pdf`` AND
    ``<output_dir>/<name>.png``. PDF is the canonical paper artefact
    (vector, scales without resampling); PNG is a high-resolution
    convenience copy for online review / Slack / email.

    The figure is closed after saving so a sequence of figure scripts
    doesn't accumulate open figures in memory. ``output_dir`` is
    created if it doesn't exist; the default ``figures/`` is the
    project's convention.

    The figure name should be domain-meaningful (``fig_bot_auroc_collapse``
    not ``fig_main`` / ``ax_0``); the schema doc is the source of
    truth for figure names.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / f"{name}.pdf")
    fig.savefig(output_dir / f"{name}.png")
    plt.close(fig)
