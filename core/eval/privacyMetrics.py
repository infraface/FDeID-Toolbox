"""
Privacy Metrics Module

This module implements comprehensive privacy metrics for face de-identification evaluation.
It measures the effectiveness of de-identification methods using face verification models.

Metrics implemented:
1. Verification Accuracy (VA↓): Rate of correctly matching de-identified faces to originals
2. Protection Success Rate (PSR↑): Percentage of faces successfully de-identified below threshold
3. True Accept Rate at FAR (TAR@FAR↓): Verification rate at fixed false accept rate
4. Equal Error Rate (EER↑): Where FAR equals FRR
5. Area Under Curve (AUC↑): Overall separability measure
6. Similarity Score Statistics: Mean, median, std of similarity distributions
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from scipy import interpolate
from sklearn.metrics import roc_curve, auc

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.identity import AdaFace, ArcFace, CosFace, FaceDetector


@dataclass
class PrivacyMetricsResult:
    """Container for privacy metrics results."""
    # Threshold-based metrics
    va: float  # Verification Accuracy (lower is better)
    psr: float  # Protection Success Rate (higher is better)
    tar_at_far_0_1: float  # TAR @ FAR=0.1% (lower is better)
    tar_at_far_1: float  # TAR @ FAR=1% (lower is better)
    tar_at_far_0_01: float  # TAR @ FAR=0.01% (lower is better)

    # Threshold-independent metrics
    eer: float  # Equal Error Rate (higher is better for privacy)
    auc: float  # Area Under ROC Curve (higher means less privacy)

    # Similarity score statistics
    mean_similarity: float  # Mean similarity score
    median_similarity: float  # Median similarity score
    std_similarity: float  # Standard deviation of similarity scores

    # Score distribution percentiles
    similarity_p25: float  # 25th percentile
    similarity_p75: float  # 75th percentile
    similarity_p95: float  # 95th percentile

    # Thresholds
    threshold_optimal: float  # Threshold at optimal accuracy
    threshold_far_0_1: float  # Threshold at FAR=0.1%
    threshold_far_1: float  # Threshold at FAR=1%
    threshold_far_0_01: float  # Threshold at FAR=0.01%

    # Additional metrics
    identity_leakage_rate: float  # Percentage with similarity > 0.9
    chance_level_protection_rate: float  # Percentage below chance level

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for easy logging."""
        return {
            'VA': self.va,
            'PSR': self.psr,
            'TAR@FAR=0.1%': self.tar_at_far_0_1,
            'TAR@FAR=1%': self.tar_at_far_1,
            'TAR@FAR=0.01%': self.tar_at_far_0_01,
            'EER': self.eer,
            'AUC': self.auc,
            'Mean_Similarity': self.mean_similarity,
            'Median_Similarity': self.median_similarity,
            'Std_Similarity': self.std_similarity,
            'Similarity_P25': self.similarity_p25,
            'Similarity_P75': self.similarity_p75,
            'Similarity_P95': self.similarity_p95,
            'Threshold_Optimal': self.threshold_optimal,
            'Threshold_FAR_0.1%': self.threshold_far_0_1,
            'Threshold_FAR_1%': self.threshold_far_1,
            'Threshold_FAR_0.01%': self.threshold_far_0_01,
            'Identity_Leakage_Rate': self.identity_leakage_rate,
            'Chance_Level_Protection_Rate': self.chance_level_protection_rate,
        }

    def __repr__(self) -> str:
        """Pretty print results."""
        return (
            f"Privacy Metrics:\n"
            f"  VA (↓): {self.va:.4f}\n"
            f"  PSR (↑): {self.psr:.4f}\n"
            f"  TAR@FAR=0.1% (↓): {self.tar_at_far_0_1:.4f}\n"
            f"  TAR@FAR=1% (↓): {self.tar_at_far_1:.4f}\n"
            f"  EER (↑): {self.eer:.4f}\n"
            f"  AUC (↑): {self.auc:.4f}\n"
            f"  Mean Similarity: {self.mean_similarity:.4f}\n"
            f"  Median Similarity: {self.median_similarity:.4f}\n"
            f"  Identity Leakage Rate: {self.identity_leakage_rate:.4f}\n"
        )


class PrivacyMetrics:
    """
    Comprehensive privacy metrics for face de-identification evaluation.

    This class computes various metrics to measure how well a de-identification
    method protects against face recognition attacks.

    Usage:
        # Single recognizer
        metrics = PrivacyMetrics(recognizer=adaface_model, detector=detector)
        result = metrics.compute(original_images, deid_images)

        # Multiple recognizers
        metrics = PrivacyMetrics(
            recognizers=[adaface, arcface, cosface],
            detector=detector
        )
        results = metrics.compute_multi_model(original_images, deid_images)
    """

    def __init__(
        self,
        recognizer: Optional[Union[AdaFace, ArcFace, CosFace]] = None,
        recognizers: Optional[List[Union[AdaFace, ArcFace, CosFace]]] = None,
        detector: Optional[FaceDetector] = None,
        device: str = 'cuda:0',
        distributed: bool = False,
        process_group: Optional[Any] = None,
    ):
        """
        Initialize PrivacyMetrics.

        Args:
            recognizer: Single face recognizer model (AdaFace, ArcFace, or CosFace)
            recognizers: List of multiple face recognizers for ensemble evaluation
            detector: Face detector (if None, assumes faces are pre-aligned)
            device: Device to run on
            distributed: Whether running in distributed mode
            process_group: Optional process group for distributed operations
        """
        self.device = device
        self.distributed = distributed
        self.process_group = process_group
        self.detector = detector

        # Handle single or multiple recognizers
        if recognizer is not None and recognizers is not None:
            raise ValueError("Specify either 'recognizer' or 'recognizers', not both")

        if recognizer is not None:
            self.recognizers = [recognizer]
            self.recognizer_names = [self._get_recognizer_name(recognizer)]
        elif recognizers is not None:
            self.recognizers = recognizers
            self.recognizer_names = [self._get_recognizer_name(r) for r in recognizers]
        else:
            raise ValueError("Must specify either 'recognizer' or 'recognizers'")

    def _get_recognizer_name(self, recognizer) -> str:
        """Get recognizer type name."""
        if isinstance(recognizer, AdaFace):
            return 'AdaFace'
        elif isinstance(recognizer, ArcFace):
            return 'ArcFace'
        elif isinstance(recognizer, CosFace):
            return 'CosFace'
        else:
            return 'Unknown'

    def compute_similarity(
        self,
        original_images: Union[List[str], List[np.ndarray], torch.Tensor],
        deid_images: Union[List[str], List[np.ndarray], torch.Tensor],
        recognizer: Union[AdaFace, ArcFace, CosFace],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Compute similarity scores between original and de-identified images.

        Args:
            original_images: Original face images (paths, arrays, or tensors)
            deid_images: De-identified face images (same format as original_images)
            recognizer: Face recognition model
            batch_size: Batch size for processing

        Returns:
            Array of similarity scores (cosine similarity, range [-1, 1])
        """
        # Extract embeddings for original images
        orig_embeddings = self._extract_embeddings(original_images, recognizer, batch_size)

        # Extract embeddings for de-identified images
        deid_embeddings = self._extract_embeddings(deid_images, recognizer, batch_size)

        # Normalize embeddings
        orig_embeddings = orig_embeddings / np.linalg.norm(orig_embeddings, axis=1, keepdims=True)
        deid_embeddings = deid_embeddings / np.linalg.norm(deid_embeddings, axis=1, keepdims=True)

        # Compute cosine similarity
        similarities = np.sum(orig_embeddings * deid_embeddings, axis=1)

        return similarities

    def _extract_embeddings(
        self,
        images: Union[List[str], List[np.ndarray], torch.Tensor],
        recognizer: Union[AdaFace, ArcFace, CosFace],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Extract embeddings from images using the recognizer.

        Args:
            images: Input images (paths, arrays, or tensors)
            recognizer: Face recognition model
            batch_size: Batch size for processing

        Returns:
            Embeddings as numpy array (N, embedding_dim)
        """
        from torchvision import transforms
        from PIL import Image

        embeddings = []

        # Prepare transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # Process in batches
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_tensors = []

            for img in batch:
                # Handle different input types
                if isinstance(img, str):
                    # Image path
                    if self.detector is not None:
                        # Detect and align
                        detections = self.detector.detect(img)
                        if detections:
                            det = detections[0]
                            aligned = recognizer.align_face(img, det.landmarks)
                            batch_tensors.append(transform(aligned))
                        else:
                            # Skip if no face detected
                            continue
                    else:
                        # Load and transform directly
                        pil_img = Image.open(img).convert('RGB')
                        batch_tensors.append(transform(pil_img))

                elif isinstance(img, np.ndarray):
                    # NumPy array
                    pil_img = Image.fromarray(img.astype(np.uint8))
                    batch_tensors.append(transform(pil_img))

                elif isinstance(img, torch.Tensor):
                    # Already a tensor
                    batch_tensors.append(img)

                else:
                    raise ValueError(f"Unsupported image type: {type(img)}")

            if not batch_tensors:
                continue

            # Stack and move to device
            batch_tensor = torch.stack(batch_tensors).to(self.device)

            # Extract embeddings
            with torch.no_grad():
                embeds = recognizer.model(batch_tensor)

                # Handle tuple output (AdaFace returns (embedding, norm))
                if isinstance(embeds, tuple):
                    embeds = embeds[0]

                embeddings.append(embeds.cpu().numpy())

        # Concatenate all batches
        if not embeddings:
            return np.array([])

        return np.vstack(embeddings)

    def compute_metrics(
        self,
        similarities: np.ndarray,
        threshold_optimal: Optional[float] = None,
        threshold_far_dict: Optional[Dict[float, float]] = None,
    ) -> PrivacyMetricsResult:
        """
        Compute privacy metrics from similarity scores.

        This assumes all pairs are genuine pairs (original vs. de-identified of same identity).
        For privacy evaluation, we want LOW verification accuracy and HIGH protection rate.

        Args:
            similarities: Array of similarity scores between original and de-identified faces
            threshold_optimal: Pre-computed optimal threshold (if None, uses 0.5 or mean)
            threshold_far_dict: Pre-computed thresholds at specific FAR values

        Returns:
            PrivacyMetricsResult object with all metrics
        """
        if len(similarities) == 0:
            raise ValueError("No similarity scores provided")

        # Create labels: all 1s since we're comparing same identities
        # In privacy context: label=1 means "should NOT match", but they do match in ground truth
        # We measure how many are INCORRECTLY verified as the same person
        labels = np.ones(len(similarities), dtype=int)

        # For privacy metrics, we need impostor distribution for FAR calculation
        # Since we don't have true impostors, we estimate thresholds from the distribution

        # Compute similarity statistics
        mean_sim = np.mean(similarities)
        median_sim = np.median(similarities)
        std_sim = np.std(similarities)
        p25 = np.percentile(similarities, 25)
        p75 = np.percentile(similarities, 75)
        p95 = np.percentile(similarities, 95)

        # Determine thresholds
        if threshold_optimal is None:
            # Use median or a reasonable default
            threshold_optimal = median_sim

        if threshold_far_dict is None:
            # Estimate thresholds from percentiles
            # These are approximations when we don't have impostor distribution
            threshold_far_dict = {
                0.0001: p95,  # Very strict (99.99th percentile)
                0.001: p75,   # Strict (75th percentile)
                0.01: median_sim,  # Moderate
            }

        # Verification Accuracy (VA): How many de-identified faces are still recognized
        # Higher threshold = stricter matching = lower VA = better privacy
        va = np.mean(similarities >= threshold_optimal)

        # Protection Success Rate (PSR): How many faces successfully avoid recognition
        # PSR = 1 - VA at the given threshold
        psr = 1.0 - va

        # TAR at specific FAR values
        tar_at_far_0_01 = np.mean(similarities >= threshold_far_dict[0.0001])
        tar_at_far_0_1 = np.mean(similarities >= threshold_far_dict[0.001])
        tar_at_far_1 = np.mean(similarities >= threshold_far_dict[0.01])

        # Compute EER and AUC (approximation without true impostor distribution)
        # We treat this as a one-class problem where all samples should be rejected
        eer, auc_score = self._compute_eer_auc_oneclass(similarities)

        # Identity leakage rate: percentage with very high similarity (>0.9)
        identity_leakage = np.mean(similarities > 0.9)

        # Chance-level protection: percentage below expected random similarity
        # For normalized embeddings, random similarity is around 0
        chance_level_protection = np.mean(similarities < 0.1)

        return PrivacyMetricsResult(
            va=va,
            psr=psr,
            tar_at_far_0_1=tar_at_far_0_1,
            tar_at_far_1=tar_at_far_1,
            tar_at_far_0_01=tar_at_far_0_01,
            eer=eer,
            auc=auc_score,
            mean_similarity=mean_sim,
            median_similarity=median_sim,
            std_similarity=std_sim,
            similarity_p25=p25,
            similarity_p75=p75,
            similarity_p95=p95,
            threshold_optimal=threshold_optimal,
            threshold_far_0_1=threshold_far_dict[0.001],
            threshold_far_1=threshold_far_dict[0.01],
            threshold_far_0_01=threshold_far_dict[0.0001],
            identity_leakage_rate=identity_leakage,
            chance_level_protection_rate=chance_level_protection,
        )

    def _compute_eer_auc_oneclass(self, similarities: np.ndarray) -> Tuple[float, float]:
        """
        Compute EER and AUC for one-class scenario (all genuine pairs).

        WARNING: This is an approximation using median-split synthetic labels.
        The resulting EER and AUC values are NOT comparable to standard two-class
        metrics computed from genuine+impostor distributions. For meaningful EER/AUC,
        use compute_thresholds_from_genuine_impostor() with both genuine and impostor scores.

        Args:
            similarities: Similarity scores

        Returns:
            Tuple of (eer, auc)
        """
        # Create synthetic binary labels based on median split
        median = np.median(similarities)
        binary_labels = (similarities <= median).astype(int)

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(binary_labels, -similarities)  # Negative for proper ordering

        # Compute AUC
        auc_score = auc(fpr, tpr)

        # Compute EER (where FPR = FNR, i.e., FPR = 1 - TPR)
        fnr = 1 - tpr
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

        return eer, auc_score

    def compute(
        self,
        original_images: Union[List[str], List[np.ndarray], torch.Tensor],
        deid_images: Union[List[str], List[np.ndarray], torch.Tensor],
        threshold_optimal: Optional[float] = None,
        threshold_far_dict: Optional[Dict[float, float]] = None,
        batch_size: int = 32,
    ) -> PrivacyMetricsResult:
        """
        Compute privacy metrics for a single recognizer.

        Args:
            original_images: Original face images
            deid_images: De-identified face images
            threshold_optimal: Pre-computed optimal threshold
            threshold_far_dict: Pre-computed thresholds at specific FAR values
            batch_size: Batch size for processing

        Returns:
            PrivacyMetricsResult object
        """
        if len(self.recognizers) > 1:
            raise ValueError("Use compute_multi_model() for multiple recognizers")

        recognizer = self.recognizers[0]

        # Compute similarities
        similarities = self.compute_similarity(
            original_images, deid_images, recognizer, batch_size
        )

        # Compute metrics
        return self.compute_metrics(similarities, threshold_optimal, threshold_far_dict)

    def compute_multi_model(
        self,
        original_images: Union[List[str], List[np.ndarray], torch.Tensor],
        deid_images: Union[List[str], List[np.ndarray], torch.Tensor],
        threshold_optimal_dict: Optional[Dict[str, float]] = None,
        threshold_far_dict: Optional[Dict[str, Dict[float, float]]] = None,
        batch_size: int = 32,
    ) -> Dict[str, PrivacyMetricsResult]:
        """
        Compute privacy metrics across multiple recognition models.

        Args:
            original_images: Original face images
            deid_images: De-identified face images
            threshold_optimal_dict: Dict mapping model name to optimal threshold
            threshold_far_dict: Dict mapping model name to FAR threshold dict
            batch_size: Batch size for processing

        Returns:
            Dictionary mapping model name to PrivacyMetricsResult
        """
        results = {}

        for recognizer, name in zip(self.recognizers, self.recognizer_names):
            print(f"Computing metrics for {name}...")

            # Get model-specific thresholds if provided
            thresh_opt = threshold_optimal_dict.get(name) if threshold_optimal_dict else None
            thresh_far = threshold_far_dict.get(name) if threshold_far_dict else None

            # Compute similarities
            similarities = self.compute_similarity(
                original_images, deid_images, recognizer, batch_size
            )

            # Compute metrics
            metrics = self.compute_metrics(similarities, thresh_opt, thresh_far)
            results[name] = metrics

        return results

    def aggregate_results(
        self,
        results: Dict[str, PrivacyMetricsResult]
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate results across multiple models.

        Args:
            results: Dictionary mapping model name to PrivacyMetricsResult

        Returns:
            Dictionary with 'mean', 'std', 'min', 'max' statistics for each metric
        """
        if not results:
            return {}

        # Convert all results to dictionaries
        all_metrics = {name: res.to_dict() for name, res in results.items()}

        # Get all metric names
        metric_names = list(next(iter(all_metrics.values())).keys())

        # Compute statistics for each metric
        aggregated = {'mean': {}, 'std': {}, 'min': {}, 'max': {}, 'worst_case': {}}

        for metric_name in metric_names:
            values = [all_metrics[model][metric_name] for model in all_metrics]

            aggregated['mean'][metric_name] = np.mean(values)
            aggregated['std'][metric_name] = np.std(values)
            aggregated['min'][metric_name] = np.min(values)
            aggregated['max'][metric_name] = np.max(values)

            # Worst case for privacy: higher VA/TAR/AUC, lower PSR/EER
            if metric_name in ['VA', 'TAR@FAR=0.1%', 'TAR@FAR=1%', 'TAR@FAR=0.01%',
                              'AUC', 'Identity_Leakage_Rate']:
                aggregated['worst_case'][metric_name] = np.max(values)
            elif metric_name in ['PSR', 'EER', 'Chance_Level_Protection_Rate']:
                aggregated['worst_case'][metric_name] = np.min(values)
            else:
                aggregated['worst_case'][metric_name] = np.mean(values)

        return aggregated


def compute_thresholds_from_genuine_impostor(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    far_values: List[float] = [0.0001, 0.001, 0.01],
) -> Tuple[float, Dict[float, float]]:
    """
    Compute optimal threshold and FAR-based thresholds from genuine and impostor distributions.

    This is the proper way to compute thresholds when you have a validation set with
    both genuine pairs and impostor pairs.

    Args:
        genuine_scores: Similarity scores for genuine pairs (same identity)
        impostor_scores: Similarity scores for impostor pairs (different identities)
        far_values: List of FAR values to compute thresholds for

    Returns:
        Tuple of (optimal_threshold, far_threshold_dict)
    """
    # Find optimal threshold (maximize accuracy)
    thresholds = np.linspace(
        min(impostor_scores.min(), genuine_scores.min()),
        max(impostor_scores.max(), genuine_scores.max()),
        1000
    )

    best_acc = 0
    optimal_threshold = 0

    for thresh in thresholds:
        tp = np.sum(genuine_scores >= thresh)
        tn = np.sum(impostor_scores < thresh)
        acc = (tp + tn) / (len(genuine_scores) + len(impostor_scores))

        if acc > best_acc:
            best_acc = acc
            optimal_threshold = thresh

    # Compute thresholds at specific FAR values
    far_thresholds = {}

    for far_val in far_values:
        # Find threshold where FAR = far_val
        # FAR = FP / (FP + TN) = FP / N_impostor
        n_impostor = len(impostor_scores)
        n_false_accepts = int(far_val * n_impostor)

        # Sort impostor scores in descending order
        sorted_impostor = np.sort(impostor_scores)[::-1]

        if n_false_accepts < len(sorted_impostor):
            threshold_at_far = sorted_impostor[n_false_accepts]
        else:
            threshold_at_far = sorted_impostor[-1]

        far_thresholds[far_val] = threshold_at_far

    return optimal_threshold, far_thresholds


def plot_similarity_distribution(
    similarities_dict: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    title: str = "Similarity Score Distribution"
):
    """
    Plot similarity score distributions for visualization.

    Args:
        similarities_dict: Dictionary mapping method name to similarity scores
        save_path: Path to save the plot
        title: Plot title
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    for method_name, similarities in similarities_dict.items():
        axes[0].hist(similarities, bins=50, alpha=0.6, label=method_name)

    axes[0].set_xlabel('Similarity Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{title} - Histogram')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Violin plot
    axes[1].violinplot(
        list(similarities_dict.values()),
        positions=range(len(similarities_dict)),
        showmeans=True,
        showmedians=True
    )
    axes[1].set_xticks(range(len(similarities_dict)))
    axes[1].set_xticklabels(list(similarities_dict.keys()), rotation=45)
    axes[1].set_ylabel('Similarity Score')
    axes[1].set_title(f'{title} - Distribution')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_metrics_comparison(
    results: Dict[str, PrivacyMetricsResult],
    save_path: Optional[str] = None,
    title: str = "Privacy Metrics Comparison"
):
    """
    Plot comparison of privacy metrics across models or methods.

    Args:
        results: Dictionary mapping model/method name to PrivacyMetricsResult
        save_path: Path to save the plot
        title: Plot title
    """
    import matplotlib.pyplot as plt

    # Extract key metrics
    methods = list(results.keys())
    metrics_to_plot = {
        'VA (↓)': [results[m].va for m in methods],
        'PSR (↑)': [results[m].psr for m in methods],
        'TAR@0.1%FAR (↓)': [results[m].tar_at_far_0_1 for m in methods],
        'EER (↑)': [results[m].eer for m in methods],
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (metric_name, values) in enumerate(metrics_to_plot.items()):
        axes[idx].bar(methods, values)
        axes[idx].set_ylabel(metric_name)
        axes[idx].set_title(metric_name)
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()
