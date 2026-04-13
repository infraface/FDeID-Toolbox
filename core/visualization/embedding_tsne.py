"""
Identity embedding t-SNE visualizer.

Generates a publication-quality 2D scatter plot of face identity embeddings
using t-SNE, showing how de-identification methods shift identity
representations. Dashed lines connect original-to-deid pairs.
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from .base import BaseVisualizer, PAPER_COLORS, PAPER_MARKERS


# Default model paths (overridable via config)
DEFAULT_RETINAFACE_MODEL = './weight/retinaface_pre_trained/Resnet50_Final.pth'
DEFAULT_ARCFACE_MODEL = './weight/ms1mv3_arcface_r100_fp16/backbone.pth'
DEFAULT_COSFACE_MODEL = './weight/glint360k_cosface_r50_fp16_0.1/backbone.pth'
DEFAULT_ADAFACE_MODEL = './weight/adaface_pre_trained/adaface_ir50_ms1mv2.ckpt'


class EmbeddingTSNEVisualizer(BaseVisualizer):
    """
    Visualizer for t-SNE projection of face identity embeddings.

    Extracts embeddings from original and de-identified images using a face
    recognition model, then projects them to 2D with t-SNE for visualization.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.original_dir = config.get('original_dir', '')
        self.deid_dirs = config.get('deid_dirs', {})
        self.model_name = config.get('model_name', 'arcface')
        self.num_images = config.get('num_images', 200)
        self.perplexity = config.get('perplexity', 30)
        self.device = config.get('device', 'cuda')
        # Allow overriding display names for methods
        self.method_display_names = config.get('method_display_names', None)

        self.retinaface_model = config.get('retinaface_model', DEFAULT_RETINAFACE_MODEL)
        self.arcface_model = config.get('arcface_model', DEFAULT_ARCFACE_MODEL)
        self.cosface_model = config.get('cosface_model', DEFAULT_COSFACE_MODEL)
        self.adaface_model = config.get('adaface_model', DEFAULT_ADAFACE_MODEL)
        self._detector = None
        self._recognizer = None
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def _load_detector(self):
        """Load RetinaFace detector on demand."""
        if self._detector is None:
            from core.identity.retinaface import FaceDetector
            self._detector = FaceDetector(
                model_path=self.retinaface_model,
                network='resnet50',
                device=self.device
            )

    def _load_recognizer(self):
        """Load face recognition model on demand."""
        if self._recognizer is not None:
            return

        from core.identity import ArcFace, CosFace, AdaFace

        if self.model_name == 'arcface':
            self._recognizer = ArcFace(
                model_path=self.arcface_model,
                num_layers=100,
                embedding_size=512,
                device=self.device
            )
        elif self.model_name == 'cosface':
            self._recognizer = CosFace(
                model_path=self.cosface_model,
                num_layers=50,
                embedding_size=512,
                device=self.device
            )
        elif self.model_name == 'adaface':
            self._recognizer = AdaFace(
                model_path=self.adaface_model,
                architecture='ir_50',
                device=self.device
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def _extract_embedding(self, img_path: str, landmarks=None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract face embedding from an image."""
        try:
            if landmarks is None:
                detections = self._detector.detect(img_path)
                if not detections:
                    return None, None
                landmarks = detections[0].landmarks

            aligned_face = self._recognizer.align_face(img_path, landmarks)
            if aligned_face is None:
                return None, None

            img_tensor = self._transform(aligned_face).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self._recognizer.model(img_tensor)
                if isinstance(embedding, tuple):
                    embedding = embedding[0]
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

            return embedding.cpu().numpy().flatten(), landmarks
        except Exception as e:
            print(f"Warning: Failed to extract embedding from {img_path}: {e}")
            return None, None

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

    def generate(self, **kwargs) -> Path:
        """Generate t-SNE embedding visualization."""
        self._load_detector()
        self._load_recognizer()

        all_deid_images = self._discover_images()
        if not all_deid_images:
            raise ValueError("No images found in deid directories")

        first_deid_dir = list(self.deid_dirs.values())[0]
        deid_data_dir = Path(self.get_deid_data_dir(first_deid_dir))

        n = min(self.num_images, len(all_deid_images))
        selected = random.sample(all_deid_images, n)
        selected.sort()

        method_keys = list(self.deid_dirs.keys())

        # Build display name mapping
        # Case 1: names cover all labels (Original + methods)
        # Case 2: names cover only method labels
        n_total = 1 + len(method_keys)
        if self.method_display_names and len(self.method_display_names) == n_total:
            orig_display = self.method_display_names[0]
            method_display = {k: self.method_display_names[i + 1]
                              for i, k in enumerate(method_keys)}
        elif self.method_display_names and len(self.method_display_names) == len(method_keys):
            orig_display = 'Original'
            method_display = {k: self.method_display_names[i]
                              for i, k in enumerate(method_keys)}
        else:
            orig_display = 'Original'
            method_display = {k: k for k in method_keys}

        all_embeddings = []
        all_labels = []
        pair_indices = []

        print(f"Extracting embeddings for {len(selected)} images...")

        for deid_img_path in selected:
            rel_path = deid_img_path.relative_to(deid_data_dir)

            orig_path = self._resolve_image_path(self.original_dir, rel_path)
            if orig_path is None:
                continue

            orig_emb, orig_lmk = self._extract_embedding(str(orig_path))
            if orig_emb is None:
                continue

            orig_idx = len(all_embeddings)
            all_embeddings.append(orig_emb)
            all_labels.append(orig_display)

            for method_key in method_keys:
                deid_dir = self.deid_dirs[method_key]
                deid_data = self.get_deid_data_dir(deid_dir)
                method_img_path = self._resolve_image_path(deid_data, rel_path)

                if method_img_path is None:
                    continue

                deid_emb, _ = self._extract_embedding(str(method_img_path), landmarks=orig_lmk)
                if deid_emb is None:
                    continue

                display_name = method_display[method_key]
                deid_idx = len(all_embeddings)
                all_embeddings.append(deid_emb)
                all_labels.append(display_name)
                pair_indices.append((orig_idx, display_name, deid_idx))

        if len(all_embeddings) < 2:
            raise ValueError("Not enough valid embeddings for t-SNE (need at least 2)")

        # Run t-SNE
        embeddings_matrix = np.array(all_embeddings)
        perplexity = min(self.perplexity, len(embeddings_matrix) - 1)
        print(f"Running t-SNE on {len(embeddings_matrix)} embeddings (perplexity={perplexity})...")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        coords_2d = tsne.fit_transform(embeddings_matrix)

        # --- Publication-quality scatter plot ---
        unique_labels = [orig_display] + [method_display[k] for k in method_keys]
        label_to_color = {label: PAPER_COLORS[i % len(PAPER_COLORS)]
                          for i, label in enumerate(unique_labels)}
        label_to_marker = {label: PAPER_MARKERS[i % len(PAPER_MARKERS)]
                           for i, label in enumerate(unique_labels)}

        fig, ax = plt.subplots(figsize=(5.5, 5.0))

        # Connecting lines (behind points)
        for orig_idx, method_name, deid_idx in pair_indices:
            ax.plot(
                [coords_2d[orig_idx, 0], coords_2d[deid_idx, 0]],
                [coords_2d[orig_idx, 1], coords_2d[deid_idx, 1]],
                color='#bbbbbb', linewidth=0.4, alpha=0.4, linestyle='--',
                zorder=1
            )

        # Scatter by label
        for label in unique_labels:
            indices = [i for i, l in enumerate(all_labels) if l == label]
            if not indices:
                continue
            ax.scatter(
                coords_2d[indices, 0], coords_2d[indices, 1],
                c=label_to_color[label],
                marker=label_to_marker[label],
                label=label,
                s=35, alpha=0.75,
                edgecolors='white', linewidths=0.3,
                zorder=2
            )

        # Clean up axes
        ax.set_xlabel('t-SNE Dimension 1', fontsize=14)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=14)

        # Remove tick labels (t-SNE values are arbitrary)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', length=0)

        # Legend
        legend = ax.legend(
            loc='best', fontsize=12,
            frameon=True, framealpha=0.9,
            edgecolor='#cccccc', fancybox=False,
            handletextpad=0.4, borderpad=0.4,
            markerscale=1.4
        )
        legend.get_frame().set_linewidth(0.6)

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color('#333333')

        plt.tight_layout()

        return self._save_figure(fig, 'embedding_tsne.png')
