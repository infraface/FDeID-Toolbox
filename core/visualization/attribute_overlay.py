"""
Attribute overlay visualizer.

Generates a publication-quality grid showing original and de-identified images
with predicted attribute labels (age, gender, expression) overlaid as text.
Attribute text is color-coded: green if preserved, red if changed.
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .base import BaseVisualizer, PAPER_COLORS


# Expression labels matching POSTER model output
EXPRESSION_LABELS = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger']

# Color-coding for preserved vs. changed attributes
COLOR_PRESERVED = '#2ca02c'  # green
COLOR_CHANGED = '#d62728'    # red


class AttributeOverlayVisualizer(BaseVisualizer):
    """
    Visualizer that overlays predicted face attributes on images.

    Shows age, gender, and expression predictions for original and
    de-identified images, color-coded to highlight preservation or change.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.original_dir = config.get('original_dir', '')
        self.deid_dirs = config.get('deid_dirs', {})
        self.attributes = config.get('attributes', ['age', 'gender', 'expression'])
        self.num_images = config.get('num_images', 5)
        self.device = config.get('device', 'cuda')
        # Allow overriding display names for methods
        self.method_display_names = config.get('method_display_names', None)

        self._fairface_predictor = None
        self._poster_model = None
        self._poster_transform = None

    def _load_fairface(self):
        """Load FairFace predictor on demand."""
        if self._fairface_predictor is None:
            from core.utility.fairface import FairFacePredictor
            self._fairface_predictor = FairFacePredictor(device=self.device)

    def _load_poster(self):
        """Load POSTER expression model on demand."""
        if self._poster_model is None:
            from core.utility.poster import load_poster_model
            from torchvision import transforms
            self._poster_model = load_poster_model(num_classes=7, device=self.device)
            self._poster_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def _predict_expression(self, img_rgb: np.ndarray) -> str:
        """Predict expression from an RGB image using POSTER."""
        self._load_poster()
        tensor = self._poster_transform(img_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self._poster_model(tensor)
            pred_idx = torch.argmax(logits, dim=1).item()
        return EXPRESSION_LABELS[pred_idx]

    def _predict_attributes(self, img_bgr: np.ndarray) -> Dict[str, str]:
        """Predict face attributes from a BGR image."""
        result = {}

        if 'age' in self.attributes or 'gender' in self.attributes:
            self._load_fairface()
            preds = self._fairface_predictor.predict(img_bgr)
            if 'age' in self.attributes:
                result['age'] = preds['age']
            if 'gender' in self.attributes:
                result['gender'] = preds['gender']

        if 'expression' in self.attributes:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            result['expression'] = self._predict_expression(img_rgb)

        return result

    def _discover_images(self) -> List[Path]:
        """Find images in the first deid directory."""
        if not self.deid_dirs:
            return []

        first_deid_dir = list(self.deid_dirs.values())[0]
        deid_data_dir = Path(self.get_deid_data_dir(first_deid_dir))

        image_paths = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            image_paths.extend(deid_data_dir.rglob(ext))

        return sorted(image_paths)

    def _resolve_image_path(self, base_dir: str, rel_path: Path) -> Optional[Path]:
        """Try to find an image at the given relative path under base_dir."""
        base = Path(base_dir)
        candidate = base / rel_path
        if candidate.exists():
            return candidate
        candidate = base / 'lfw-deepfunneled' / 'lfw-deepfunneled' / rel_path
        if candidate.exists():
            return candidate
        return None

    def _build_attr_label(self, attrs: Dict[str, str],
                          orig_attrs: Optional[Dict[str, str]] = None
                          ) -> List[Tuple[str, str]]:
        """
        Build list of (text, color) tuples for attribute annotation.

        If orig_attrs is provided, colors are green/red based on match.
        Otherwise, all text is black.
        """
        parts = []
        for key in ['age', 'gender', 'expression']:
            if key not in attrs:
                continue
            val = attrs[key]
            label = f"Age:{val}" if key == 'age' else val

            if orig_attrs is not None and key in orig_attrs:
                color = COLOR_PRESERVED if val == orig_attrs[key] else COLOR_CHANGED
            else:
                color = '#333333'
            parts.append((label, color))
        return parts

    def generate(self, **kwargs) -> Path:
        """Generate attribute overlay comparison grid with horizontal layout.

        Layout: rows = [Original, Method1, Method2, ...], columns = images.
        Row labels are displayed on the left side of each row.
        """
        all_deid_images = self._discover_images()
        if not all_deid_images:
            raise ValueError("No images found in deid directories")

        first_deid_dir = list(self.deid_dirs.values())[0]
        deid_data_dir = Path(self.get_deid_data_dir(first_deid_dir))

        n = min(self.num_images, len(all_deid_images))
        selected = random.sample(all_deid_images, n)
        selected.sort()

        method_keys = list(self.deid_dirs.keys())
        # Use display names from argument if provided
        # Case 1: names cover all rows (Original + methods)
        # Case 2: names cover only method rows
        n_total_rows = 1 + len(method_keys)
        if self.method_display_names and len(self.method_display_names) == n_total_rows:
            row_labels = list(self.method_display_names)
        elif self.method_display_names and len(self.method_display_names) == len(method_keys):
            row_labels = ['Original'] + list(self.method_display_names)
        else:
            row_labels = ['Original'] + method_keys

        # Horizontal layout: rows=methods, columns=images
        n_rows = n_total_rows
        n_cols = len(selected)

        # Each cell: image + vertically stacked text below
        cell_w = 2.4
        cell_h = 3.2  # taller to accommodate stacked attribute lines
        label_margin = 0.6  # space for vertical row labels on the left
        fig_w = cell_w * n_cols + label_margin
        fig_h = cell_h * n_rows + 0.2
        fig = plt.figure(figsize=(fig_w, fig_h))
        left = label_margin / fig_w
        gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.05, hspace=0.35,
                      top=0.98, bottom=0.02, left=left, right=0.98)

        # Pre-compute original attributes for each selected image
        orig_data = []
        for deid_img_path in selected:
            rel_path = deid_img_path.relative_to(deid_data_dir)
            orig_path = self._resolve_image_path(self.original_dir, rel_path)
            orig_bgr, orig_rgb = None, None
            orig_attrs = {}
            if orig_path is not None:
                orig_bgr = cv2.imread(str(orig_path))
            if orig_bgr is not None:
                orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
                orig_attrs = self._predict_attributes(orig_bgr)
            orig_data.append((rel_path, orig_bgr, orig_rgb, orig_attrs))

        # --- Original row ---
        for col_idx, (rel_path, orig_bgr, orig_rgb, orig_attrs) in enumerate(orig_data):
            ax = fig.add_subplot(gs[0, col_idx])
            self._show_image(ax, orig_rgb)
            if col_idx == 0:
                ax.set_ylabel(row_labels[0], fontsize=14, fontweight='bold',
                              rotation=90, labelpad=10, va='center')
            orig_parts = self._build_attr_label(orig_attrs)
            self._annotate_attrs(ax, orig_parts)

        # --- Method rows ---
        for row_idx, method_key in enumerate(method_keys):
            deid_dir = self.deid_dirs[method_key]
            deid_data = self.get_deid_data_dir(deid_dir)

            for col_idx, (rel_path, orig_bgr, orig_rgb, orig_attrs) in enumerate(orig_data):
                method_img_path = self._resolve_image_path(deid_data, rel_path)

                deid_bgr, deid_rgb = None, None
                deid_attrs = {}
                if method_img_path is not None:
                    deid_bgr = cv2.imread(str(method_img_path))
                if deid_bgr is not None:
                    deid_rgb = cv2.cvtColor(deid_bgr, cv2.COLOR_BGR2RGB)
                    deid_attrs = self._predict_attributes(deid_bgr)

                ax = fig.add_subplot(gs[row_idx + 1, col_idx])
                self._show_image(ax, deid_rgb if deid_rgb is not None else orig_rgb)
                if col_idx == 0:
                    ax.set_ylabel(row_labels[row_idx + 1], fontsize=14,
                                  fontweight='bold', rotation=90,
                                  labelpad=10, va='center')

                deid_parts = self._build_attr_label(deid_attrs, orig_attrs)
                self._annotate_attrs(ax, deid_parts)

        return self._save_figure(fig, 'attribute_overlay.png')

    @staticmethod
    def _show_image(ax, img: Optional[np.ndarray]):
        """Display an image with thin border."""
        if img is not None:
            ax.imshow(img)
        else:
            ax.imshow(np.ones((112, 112, 3), dtype=np.uint8) * 240)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color('#333333')

    @staticmethod
    def _annotate_attrs(ax, parts: List[Tuple[str, str]]):
        """
        Place attribute labels vertically stacked below the image.

        Each attribute is placed on its own line to avoid horizontal overlap.
        """
        if not parts:
            return

        line_spacing = 0.065
        y_start = -0.05
        for i, (text, color) in enumerate(parts):
            ax.text(0.5, y_start - i * line_spacing, text,
                    transform=ax.transAxes,
                    fontsize=10, color=color, fontweight='semibold',
                    ha='center', va='top', family='sans-serif')
