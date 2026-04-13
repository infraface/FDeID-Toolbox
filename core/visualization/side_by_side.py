"""
Side-by-side comparison visualizer.

Generates a clean grid showing original images alongside de-identified results
from multiple methods for qualitative comparison in publication figures.
"""

import random
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .base import BaseVisualizer


class SideBySideVisualizer(BaseVisualizer):
    """
    Visualizer for side-by-side comparison of de-identification methods.

    Creates a clean grid where rows are images and columns are
    [Original, Method1, Method2, ...] with thin borders and no distracting
    elements, suitable for direct inclusion in scientific papers.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.original_dir = config.get('original_dir', '')
        self.deid_dirs = config.get('deid_dirs', {})
        self.num_images = config.get('num_images', 5)
        self.image_names = config.get('image_names', None)
        self.show_bbox = config.get('show_bbox', False)
        # Allow overriding display names for methods
        self.method_display_names = config.get('method_display_names', None)

    def _discover_images(self) -> List[Path]:
        """Find images in the first deid directory and match with originals."""
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

    def generate(self, **kwargs) -> Path:
        """Generate side-by-side comparison grid with horizontal layout.

        Layout: rows = [Original, Method1, Method2, ...], columns = images.
        Row labels are displayed on the left side of each row.
        """
        all_deid_images = self._discover_images()
        if not all_deid_images:
            raise ValueError("No images found in deid directories")

        first_deid_dir = list(self.deid_dirs.values())[0]
        deid_data_dir = Path(self.get_deid_data_dir(first_deid_dir))

        # Select images
        if self.image_names:
            selected = [p for p in all_deid_images if p.name in self.image_names]
            if not selected:
                raise ValueError("None of the requested image names found")
        else:
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

        # Load optional face detector for bounding boxes
        detector = None
        if self.show_bbox:
            try:
                from core.identity.retinaface import FaceDetector
                retinaface_model = self.config.get('retinaface_model',
                    './weight/retinaface_pre_trained/Resnet50_Final.pth')
                device = self.config.get('device', 'cuda')
                detector = FaceDetector(
                    model_path=retinaface_model,
                    network='resnet50',
                    device=device
                )
            except Exception as e:
                print(f"Warning: Could not load face detector for bbox overlay: {e}")

        # Create figure with horizontal layout
        cell_size = 2.0
        label_margin = 1.4  # space for row labels on the left
        fig_w = cell_size * n_cols + label_margin
        fig_h = cell_size * n_rows + 0.2
        fig = plt.figure(figsize=(fig_w, fig_h))
        left = label_margin / fig_w
        gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.03, hspace=0.05,
                      top=0.98, bottom=0.02, left=left, right=0.98)

        for col_idx, deid_img_path in enumerate(selected):
            rel_path = deid_img_path.relative_to(deid_data_dir)

            # --- Original row ---
            orig_path = self._resolve_image_path(self.original_dir, rel_path)
            orig_img = self._load_rgb(orig_path)
            if detector is not None and orig_img is not None:
                orig_img = self._draw_bboxes(orig_img, detector)

            ax = fig.add_subplot(gs[0, col_idx])
            self._show_image(ax, orig_img)
            if col_idx == 0:
                ax.set_ylabel(row_labels[0], fontsize=14, fontweight='bold',
                              rotation=0, labelpad=70, va='center')

            # --- Method rows ---
            for row_idx, method_key in enumerate(method_keys):
                deid_dir = self.deid_dirs[method_key]
                deid_data = self.get_deid_data_dir(deid_dir)
                method_img_path = self._resolve_image_path(deid_data, rel_path)
                img = self._load_rgb(method_img_path)
                if detector is not None and img is not None:
                    img = self._draw_bboxes(img, detector)

                ax = fig.add_subplot(gs[row_idx + 1, col_idx])
                self._show_image(ax, img if img is not None else orig_img)
                if col_idx == 0:
                    ax.set_ylabel(row_labels[row_idx + 1], fontsize=14,
                                  fontweight='bold', rotation=0,
                                  labelpad=70, va='center')

        return self._save_figure(fig, 'side_by_side.png')

    @staticmethod
    def _load_rgb(path: Optional[Path]) -> Optional[np.ndarray]:
        """Load image as RGB numpy array."""
        if path is None:
            return None
        img = cv2.imread(str(path))
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _show_image(ax, img: Optional[np.ndarray]):
        """Display an image in an axis with thin black border and no ticks."""
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
    def _draw_bboxes(img: np.ndarray, detector) -> np.ndarray:
        """Draw face bounding boxes on an image."""
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        try:
            detections = detector.detect(img_bgr)
            for det in detections:
                bbox = det.bbox.astype(int)
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                              (0, 200, 0), 2)
        except Exception:
            pass
        return img
