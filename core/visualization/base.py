"""
Base class for visualization utilities.

Provides common functionality for saving figures, loading de-identification
configs, resolving data directories, and applying publication-quality style.
"""

import os
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl


# Nature-style qualitative palette (Wong 2011, colorblind-safe and print-safe).
# This is the palette recommended for Nature Methods figures.
PAPER_COLORS = [
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#009E73',  # bluish green
    '#E69F00',  # orange
    '#CC79A7',  # reddish purple
    '#56B4E9',  # sky blue
    '#999999',  # gray
    '#000000',  # black
]

PAPER_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h']


def apply_paper_style():
    """
    Apply Nature-style matplotlib defaults.

    Follows the Nature figure convention: a clean sans-serif typeface
    (Helvetica/Arial), outward ticks, only the left and bottom spines, no
    bold titles, and a colorblind-safe palette. Tuned to stay legible after
    the figures are scaled down to one- or two-column width in the manuscript.
    """
    style = {
        # Font: Nature uses Helvetica/Arial sans-serif
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 13,
        'mathtext.fontset': 'dejavusans',
        'axes.unicode_minus': False,

        # Axes: only left/bottom spines, thin and dark gray
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#222222',
        'axes.labelsize': 15,
        'axes.titlesize': 15,
        'axes.titleweight': 'normal',
        'axes.labelweight': 'normal',
        'axes.labelcolor': '#222222',
        'axes.spines.top': False,
        'axes.spines.right': False,

        # Ticks: outward, Nature convention
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.color': '#222222',
        'ytick.color': '#222222',
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'xtick.major.pad': 3,
        'ytick.major.pad': 3,

        # Legend: light, borderless-feeling
        'legend.fontsize': 12,
        'legend.frameon': False,
        'legend.framealpha': 0.0,
        'legend.handlelength': 1.4,
        'legend.handletextpad': 0.5,
        'legend.columnspacing': 1.2,

        # Figure
        'figure.facecolor': 'white',
        'figure.edgecolor': 'white',
        'figure.dpi': 150,
        'figure.titlesize': 16,
        'figure.titleweight': 'normal',

        # Saving
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,

        # Lines
        'lines.linewidth': 1.6,
        'lines.markersize': 5,

        # Grid: faint solid hairlines
        'grid.linewidth': 0.5,
        'grid.alpha': 0.4,
        'grid.color': '#dddddd',
        'grid.linestyle': '-',
    }
    mpl.rcParams.update(style)


class BaseVisualizer(ABC):
    """
    Abstract base class for all visualization utilities.

    All visualizers (side-by-side, attribute overlay, embedding t-SNE, radar chart)
    should inherit from this class and implement the generate() method.
    """

    def __init__(self, config: Dict):
        """
        Initialize visualizer with configuration.

        Args:
            config: Dictionary containing visualizer-specific parameters.
                    Common keys: save_dir, dpi
        """
        self.config = config
        self.save_dir = Path(config.get('save_dir', '.'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = config.get('dpi', 300)

        # Apply publication style on init
        apply_paper_style()

    @abstractmethod
    def generate(self, **kwargs) -> Path:
        """
        Generate visualization, save to disk, and return the output path.

        Returns:
            Path to the saved visualization file.
        """
        pass

    def _save_figure(self, fig: plt.Figure, filename: str) -> Path:
        """
        Save a matplotlib figure as both PNG and PDF.

        Args:
            fig: Matplotlib figure to save.
            filename: Output filename (e.g. 'side_by_side.png').

        Returns:
            Path to the saved PNG file.
        """
        png_path = self.save_dir / filename
        fig.savefig(str(png_path), dpi=self.dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

        # Also save PDF for vector graphics (paper-ready)
        pdf_path = self.save_dir / filename.replace('.png', '.pdf')
        fig.savefig(str(pdf_path), bbox_inches='tight',
                    facecolor='white', edgecolor='none')

        plt.close(fig)
        print(f"Visualization saved to: {png_path}")
        print(f"PDF version saved to:   {pdf_path}")
        return png_path

    @staticmethod
    def load_deid_config(deid_dir: str) -> Optional[Dict]:
        """Load de-identification configuration if config.yaml exists."""
        config_path = os.path.join(deid_dir, 'config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return None

    @staticmethod
    def get_deid_data_dir(deid_dir: str) -> str:
        """Get the actual data directory (handles both old and new structure)."""
        data_subdir = os.path.join(deid_dir, 'data')
        if os.path.exists(data_subdir) and os.path.isdir(data_subdir):
            return data_subdir
        return deid_dir
