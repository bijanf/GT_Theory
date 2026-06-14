"""Matplotlib rcParams matching Springer Nature submission guidelines.

Sans-serif body (Helvetica/Arial), 6-7 pt text, vector PDF output with
TrueType embedded fonts (pdf.fonttype=42), RGB color, and Nature 1-column
(88 mm) and 2-column (180 mm) widths.

Reference: Springer Nature, "Guide to Preparing Final Artwork".
"""

from __future__ import annotations

import matplotlib as mpl

# Nature column widths in inches.
NATURE_1COL_INCH: float = 3.46  # 88 mm
NATURE_2COL_INCH: float = 7.09  # 180 mm


def apply_nature_style() -> None:
    """Set the global rcParams to Nature-spec for the current process."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            # Nimbus Sans is the URW Helvetica metric-clone; Liberation
            # Sans is the open Arial metric-clone. Both are pre-installed
            # on typical Linux scientific images; either satisfies the
            # Springer Nature "Helvetica or Arial" requirement without
            # licensed-font shipping. DejaVu Sans is the matplotlib
            # default fallback and is intentionally listed last.
            "font.sans-serif": [
                "Nimbus Sans",
                "Helvetica",
                "Liberation Sans",
                "Arial",
                "DejaVu Sans",
            ],
            # Bump base size from 6 to 7 pt: 6 pt mathtext subscripts
            # render at 4.2 pt (below Nature's 5 pt floor); 7 pt gets
            # subscripts to 4.9 pt -- borderline but tolerated; axis-label
            # subscripts at 8 pt base render at 5.6 pt and are compliant.
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            # Force mathtext to also use Nimbus Sans (Helvetica clone)
            # rather than the matplotlib default (cm/DejaVu Sans), so that
            # the rendered PDF embeds Nimbus everywhere -- no DejaVu
            # fallback for italic glyphs in $T_s$, $\Delta T$, etc.
            "mathtext.fontset": "custom",
            "mathtext.rm": "Nimbus Sans",
            "mathtext.it": "Nimbus Sans:italic",
            "mathtext.bf": "Nimbus Sans:bold",
            "mathtext.sf": "Nimbus Sans",
            "mathtext.cal": "Nimbus Sans:italic",
            "mathtext.default": "regular",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.format": "pdf",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.0,
            "patch.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
        }
    )
