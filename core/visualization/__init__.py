"""
Visualization Module

Provides visualization utilities for qualitative analysis and method comparison:
- SideBySideVisualizer: Grid comparison of original vs. de-identified images
- AttributeOverlayVisualizer: Attribute predictions overlaid on face images
- EmbeddingTSNEVisualizer: t-SNE projection of identity embeddings
- RadarChartVisualizer: Multi-dimensional radar/spider charts for metric comparison
"""

from .base import BaseVisualizer
from .side_by_side import SideBySideVisualizer
from .attribute_overlay import AttributeOverlayVisualizer
from .embedding_tsne import EmbeddingTSNEVisualizer
from .radar_chart import RadarChartVisualizer


def get_visualizer(config: dict) -> BaseVisualizer:
    """
    Factory function to create a visualizer from configuration.

    Args:
        config: Dictionary containing at minimum a 'viz_type' key.
                Supported types: 'side_by_side', 'attribute_overlay',
                'embedding_tsne', 'radar_chart'.

    Returns:
        An initialized visualizer instance.

    Raises:
        ValueError: If viz_type is not recognized.
    """
    viz_type = config.get('viz_type', '')

    if viz_type == 'side_by_side':
        return SideBySideVisualizer(config)
    elif viz_type == 'attribute_overlay':
        return AttributeOverlayVisualizer(config)
    elif viz_type == 'embedding_tsne':
        return EmbeddingTSNEVisualizer(config)
    elif viz_type == 'radar_chart':
        return RadarChartVisualizer(config)
    else:
        raise ValueError(
            f"Unknown viz_type: '{viz_type}'. "
            f"Supported types: side_by_side, attribute_overlay, embedding_tsne, radar_chart"
        )


__all__ = [
    'BaseVisualizer',
    'SideBySideVisualizer',
    'AttributeOverlayVisualizer',
    'EmbeddingTSNEVisualizer',
    'RadarChartVisualizer',
    'get_visualizer',
]
