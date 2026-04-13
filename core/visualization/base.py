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


# Publication-quality color palette (colorblind-friendly, print-safe)
PAPER_COLORS = [
    '#1f77b4',  # blue
    '#d62728',  # red
    '#2ca02c',  # green
    '#ff7f0e',  # orange
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
]

PAPER_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h']


def apply_paper_style():
    """
    Apply publication-quality matplotlib style (CVPR/ECCV/NeurIPS convention).

    Sets serif fonts, proper tick directions, clean spines, and high-quality
    rendering defaults suitable for camera-ready figures.
    """
    style = {
        # Font
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 12,
        'mathtext.fontset': 'stix',

        # Axes
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#333333',
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'normal',
        'axes.spines.top': True,
        'axes.spines.right': True,

        # Ticks
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'xtick.major.pad': 4,
        'ytick.major.pad': 4,

        # Legend
        'legend.fontsize': 12,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#cccccc',
        'legend.fancybox': False,
        'legend.handlelength': 1.5,

        # Figure
        'figure.facecolor': 'white',
        'figure.edgecolor': 'white',
        'figure.dpi': 150,
        'figure.titlesize': 15,
        'figure.titleweight': 'bold',

        # Saving
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,

        # Lines
        'lines.linewidth': 1.5,
        'lines.markersize': 5,

        # Grid
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'grid.color': '#cccccc',
        'grid.linestyle': '--',
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
