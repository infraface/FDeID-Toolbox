"""
Radar (spider) chart visualizer.

Generates publication-quality multi-dimensional radar charts for comparing
de-identification methods across privacy, utility, and quality metrics.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .base import BaseVisualizer, PAPER_COLORS


# Default utility preservation dimensions (6 dimensions)
DEFAULT_UTILITY_METRICS = ['Age', 'Gender', 'Ethnicity', 'Expression', 'Landmark', 'rPPG']

# Default metrics and their directionality
# True means higher is better, False means lower is better
DEFAULT_METRIC_DIRECTIONS = {
    'Age': False,        # Age MAE - lower is better
    'Gender': True,      # Gender accuracy - higher is better
    'Ethnicity': True,   # Ethnicity accuracy - higher is better
    'Expression': True,  # Expression accuracy - higher is better
    'Landmark': False,   # Landmark NME - lower is better
    'rPPG': False,       # rPPG HR-MAE - lower is better
}


class RadarChartVisualizer(BaseVisualizer):
    """
    Visualizer for multi-dimensional radar/spider charts.

    Compares de-identification methods across multiple metrics on a single
    radar chart with normalized axes and per-method polygons.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.results = config.get('results', {})
        self.results_dirs = config.get('results_dirs', [])
        self.method_names = config.get('method_names', [])
        self.metrics = config.get('metrics', DEFAULT_UTILITY_METRICS)
        self.metric_directions = config.get('metric_directions', DEFAULT_METRIC_DIRECTIONS)
        # Reference baseline (e.g., Original) for normalization
        self.reference = config.get('radar_reference', {})
        # Support manually set values: dict of {method_name: {metric: value}}
        radar_values = config.get('radar_values', {})
        if radar_values:
            self.results.update(radar_values)

    def _load_results_from_dirs(self):
        """Load results from JSON files and populate self.results."""
        for i, results_path in enumerate(self.results_dirs):
            path = Path(results_path)
            if not path.exists():
                print(f"Warning: Results file not found: {results_path}")
                continue

            with open(path, 'r') as f:
                data = json.load(f)

            if i < len(self.method_names):
                method_name = self.method_names[i]
            else:
                deid_config = data.get('deid_config', {})
                if deid_config:
                    method_name = deid_config.get('method_name', f'method_{i}')
                else:
                    method_name = f'method_{i}'

            eval_results = data.get('evaluation_results', {})
            metric_dict = self._extract_metrics(eval_results)

            if metric_dict:
                self.results[method_name] = metric_dict

    def _extract_metrics(self, eval_results: Dict) -> Dict[str, float]:
        """Extract metrics from evaluation results JSON structure."""
        metrics = {}

        # Privacy metrics
        for model in ['arcface', 'cosface', 'adaface']:
            if model in eval_results:
                model_results = eval_results[model]
                if 'protection_success_rate' in model_results:
                    metrics['PSR'] = model_results['protection_success_rate']
                    break

        # Quality metrics
        for metric_key in ['psnr', 'ssim', 'lpips', 'niqe_deidentified']:
            if metric_key in eval_results:
                val = eval_results[metric_key]
                if isinstance(val, dict) and 'mean' in val:
                    label = metric_key.upper().replace('_DEIDENTIFIED', '')
                    metrics[label] = val['mean']
                elif isinstance(val, (int, float)):
                    metrics[metric_key.upper()] = val

        if 'fid' in eval_results:
            metrics['FID'] = eval_results['fid']

        # Utility metrics
        for key in ['Age', 'Gender', 'Ethnicity', 'Expression', 'Landmark',
                     'age_accuracy', 'gender_accuracy', 'ethnicity_accuracy',
                     'expression_accuracy', 'landmark_accuracy']:
            if key in eval_results:
                val = eval_results[key]
                if isinstance(val, dict) and 'accuracy' in val:
                    label = key.replace('_accuracy', '').capitalize()
                    metrics[label] = val['accuracy']
                elif isinstance(val, (int, float)):
                    metrics[key] = val

        if 'rPPG' in eval_results or 'rppg' in eval_results:
            val = eval_results.get('rPPG', eval_results.get('rppg'))
            if isinstance(val, dict):
                metrics['rPPG'] = val.get('correlation', val.get('mean', 0))
            elif isinstance(val, (int, float)):
                metrics['rPPG'] = val

        return metrics

    def _normalize_metrics(self) -> Dict[str, Dict[str, float]]:
        """Normalize metric values to [0, 1] range.

        When a reference baseline (e.g., Original) is provided, normalization
        computes the utility preservation ratio relative to the reference:
          - Higher-is-better: score = method_value / reference_value
          - Lower-is-better:  score = reference_value / method_value
        This gives 1.0 for perfect preservation and ~0 for severe degradation.

        Without a reference, falls back to min-max normalization.
        """
        method_names = list(self.results.keys())
        available_metrics = []
        for metric in self.metrics:
            if any(metric in self.results[m] for m in method_names):
                available_metrics.append(metric)

        if not available_metrics:
            return {}

        # Reference-based normalization (preferred)
        if self.reference:
            normalized = {}
            for method in method_names:
                normalized[method] = {}
                for metric in available_metrics:
                    if metric not in self.results[method]:
                        normalized[method][metric] = 0.0
                        continue

                    val = self.results[method][metric]
                    ref = self.reference.get(metric)
                    if ref is None or ref == 0:
                        normalized[method][metric] = 1.0 if val == 0 else 0.0
                        continue

                    higher_is_better = self.metric_directions.get(metric, True)
                    if higher_is_better:
                        norm_val = val / ref
                    else:
                        norm_val = ref / val if val != 0 else 1.0

                    normalized[method][metric] = min(max(norm_val, 0.0), 1.0)
            return normalized

        # Fallback: min-max normalization
        metric_ranges = {}
        for metric in available_metrics:
            values = [self.results[m][metric] for m in method_names if metric in self.results[m]]
            if values:
                metric_ranges[metric] = (min(values), max(values))

        normalized = {}
        for method in method_names:
            normalized[method] = {}
            for metric in available_metrics:
                if metric not in self.results[method]:
                    normalized[method][metric] = 0.0
                    continue

                val = self.results[method][metric]
                min_val, max_val = metric_ranges[metric]

                if max_val == min_val:
                    norm_val = 1.0
                else:
                    norm_val = (val - min_val) / (max_val - min_val)

                higher_is_better = self.metric_directions.get(metric, True)
                if not higher_is_better:
                    norm_val = 1.0 - norm_val

                normalized[method][metric] = norm_val

        return normalized

    def generate(self, **kwargs) -> Path:
        """Generate radar chart visualization with 6 utility preservation dimensions."""
        if self.results_dirs and not self.results:
            self._load_results_from_dirs()

        if not self.results:
            raise ValueError(
                "No results provided. Use 'results' dict, 'results_dirs' list, "
                "or 'radar_values' dict for manual values."
            )

        normalized = self._normalize_metrics()
        if not normalized:
            raise ValueError("No valid metrics found to plot")

        method_names = list(normalized.keys())
        # Use display names if provided
        method_display = self.config.get('method_display_names', None)
        if not method_display or len(method_display) != len(method_names):
            method_display = method_names

        available_metrics = []
        for metric in self.metrics:
            if any(metric in normalized[m] for m in method_names):
                available_metrics.append(metric)

        if not available_metrics:
            raise ValueError("No matching metrics found in results")

        n_metrics = len(available_metrics)

        # Compute angles
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]

        # --- Publication-quality radar chart ---
        fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))

        # Draw concentric reference rings
        ring_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
        for level in ring_levels:
            ring_angles = np.linspace(0, 2 * np.pi, 100)
            ax.plot(ring_angles, [level] * 100, color='#dddddd', linewidth=0.5,
                    linestyle='-', zorder=0)

        # Draw radial grid lines
        for angle in angles[:-1]:
            ax.plot([angle, angle], [0, 1.05], color='#dddddd', linewidth=0.5,
                    linestyle='-', zorder=0)

        # Plot each method
        for idx, method in enumerate(method_names):
            values = [normalized[method].get(m, 0) for m in available_metrics]
            values += values[:1]
            color = PAPER_COLORS[idx % len(PAPER_COLORS)]
            label = method_display[idx]

            ax.plot(angles, values, '-', linewidth=1.8, label=label,
                    color=color, zorder=3)
            ax.scatter(angles[:-1], values[:-1], color=color, s=30,
                       edgecolors='white', linewidths=0.5, zorder=4)
            ax.fill(angles, values, alpha=0.08, color=color, zorder=1)

        # Axis labels at each spoke
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(available_metrics, fontsize=13, fontweight='medium')

        # Radial ticks
        ax.set_ylim(0, 1.08)
        ax.set_yticks(ring_levels)
        ax.set_yticklabels([f'{v:.1f}' for v in ring_levels],
                           fontsize=9, color='#888888')
        ax.yaxis.set_tick_params(labelsize=9)

        # Remove default grid
        ax.grid(False)

        # Spine styling
        ax.spines['polar'].set_visible(False)

        # Legend
        legend = ax.legend(
            loc='upper right', bbox_to_anchor=(1.35, 1.10),
            fontsize=12, frameon=True, framealpha=0.9,
            edgecolor='#cccccc', fancybox=False,
            handlelength=1.8, handletextpad=0.5, borderpad=0.5
        )
        legend.get_frame().set_linewidth(0.6)

        plt.tight_layout(pad=1.5)

        return self._save_figure(fig, 'radar_chart.png')
