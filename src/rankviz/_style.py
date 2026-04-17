"""Publication-quality style defaults for rankviz.

Provides a consistent visual language across all rankviz plots:
300 DPI, thin spines, no embedded titles, colourblind-safe palette.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paul Tol qualitative palette (colourblind-safe)
# https://personal.sron.nl/~pault/data/colourschemes.pdf
# ---------------------------------------------------------------------------

PALETTE_HIGHLIGHT = [
    "#EE6677",  # red / rose
    "#228833",  # green
    "#4477AA",  # blue
    "#CCBB44",  # yellow
    "#66CCEE",  # cyan
    "#AA3377",  # purple
    "#BBBBBB",  # grey (fallback)
]

CORPUS_COLOUR = "#CCCCCC"
CORPUS_COLOUR_DARK = "#999999"

# ---------------------------------------------------------------------------
# Font fallback chain: Helvetica -> Arial -> DejaVu Sans
# ---------------------------------------------------------------------------

_FONT_CANDIDATES = ["Helvetica", "Arial", "DejaVu Sans"]


def _resolve_font() -> str:
    """Return the first available font from the fallback chain."""
    available = {f.name for f in mpl.font_manager.fontManager.ttflist}
    for candidate in _FONT_CANDIDATES:
        if candidate in available:
            return candidate
    return "DejaVu Sans"


# ---------------------------------------------------------------------------
# rcParams for publication-quality figures
# ---------------------------------------------------------------------------

def _base_rcparams() -> dict:
    """Return rcParams dict for publication-quality output."""
    font = _resolve_font()
    return {
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": [font],
        "font.size": 9,
        # Axes
        "axes.linewidth": 0.5,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Ticks
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.3,
        "ytick.minor.width": 0.3,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        # Legend
        "legend.fontsize": 8,
        "legend.frameon": False,
        # Figure
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # Lines
        "lines.linewidth": 1.0,
    }


@contextmanager
def style() -> Generator[None, None, None]:
    """Context manager that temporarily applies rankviz publication defaults.

    Usage::

        with rankviz.style():
            fig = rc.plot()
            fig.savefig("figure.pdf")
    """
    with mpl.rc_context(_base_rcparams()):
        yield


def apply_style() -> None:
    """Permanently apply rankviz defaults to the current matplotlib session."""
    mpl.rcParams.update(_base_rcparams())


def get_highlight_colours(n: int) -> list[str]:
    """Return *n* distinct highlight colours from the Paul Tol palette.

    Cycles if *n* exceeds the palette length.
    """
    return [PALETTE_HIGHLIGHT[i % len(PALETTE_HIGHLIGHT)] for i in range(n)]
