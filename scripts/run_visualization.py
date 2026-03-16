#!/usr/bin/env python3
"""
Visualization CLI Entry Point.

Generates qualitative visualizations for comparing de-identification methods:
- side_by_side: Grid comparison of original vs. de-identified images
- attribute_overlay: Face attribute predictions overlaid on images
- embedding_tsne: t-SNE projection of identity embeddings
- radar_chart: Multi-dimensional radar chart of evaluation metrics

Usage:
    # Side-by-side comparison (horizontal layout)
    python scripts/run_visualization.py \
        --viz_type side_by_side \
        --original_dir /path/to/original \
        --deid_dirs "blur=/path/to/blur,ciagan=/path/to/ciagan" \
        --method_names "Gaussian Blur" "CIAGAN" \
        --num_images 5 \
        --save_dir runs/vis/

    # Attribute overlay
    python scripts/run_visualization.py \
        --viz_type attribute_overlay \
        --original_dir /path/to/original \
        --deid_dirs "blur=/path/to/blur" \
        --attributes age,gender,expression \
        --num_images 5

    # Embedding t-SNE
    python scripts/run_visualization.py \
        --viz_type embedding_tsne \
        --original_dir /path/to/original \
        --deid_dirs "blur=/path/to/blur,ciagan=/path/to/ciagan" \
        --model_name arcface \
        --num_images 50

    # Radar chart from evaluation results
    python scripts/run_visualization.py \
        --viz_type radar_chart \
        --results_dirs /path/to/blur/results.json,/path/to/ciagan/results.json \
        --method_names blur,ciagan \
        --metrics Age,Gender,Ethnicity,Expression,Landmark,rPPG

    # Radar chart with manual values (reference-based normalization)
    python scripts/run_visualization.py \
        --viz_type radar_chart \
        --radar_values "k-Same-Select=17.81,59.16,20.49,15.55,0.3973,32.56;DeID-rPPG=10.78,94.03,70.28,69.38,0.3627,0.39" \
        --radar_reference "10.18,94.39,71.99,70.54,0.3601,0.39"
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.visualization import get_visualizer
from core.config_utils import load_config_into_args


def parse_deid_dirs(deid_dirs_str: str) -> dict:
    """
    Parse comma-separated 'name=path' pairs into a dictionary.

    Args:
        deid_dirs_str: e.g. "blur=/path/to/blur,ciagan=/path/to/ciagan"

    Returns:
        Dict mapping method names to paths.
    """
    result = {}
    if not deid_dirs_str:
        return result

    for pair in deid_dirs_str.split(','):
        pair = pair.strip()
        if '=' in pair:
            name, path = pair.split('=', 1)
            result[name.strip()] = path.strip()
        else:
            # Use directory basename as name
            result[Path(pair).name] = pair.strip()

    return result


def flatten_list_arg(values):
    """
    Flatten a list that may contain comma-separated items.

    Handles both space-separated (nargs='+') and comma-separated inputs:
        ['a,b', 'c'] -> ['a', 'b', 'c']
        ['a', 'b', 'c'] -> ['a', 'b', 'c']
        ['a,b,c'] -> ['a', 'b', 'c']
    """
    result = []
    for v in values:
        for item in v.split(','):
            item = item.strip()
            if item:
                result.append(item)
    return result


def parse_radar_values(radar_str: str, metrics: list) -> dict:
    """
    Parse manually set radar chart values string.

    Format: "Method1=v1,v2,...,vN;Method2=v1,v2,...,vN"
    Values correspond to metrics in order.

    Args:
        radar_str: e.g. "Blur=80,90,85,75,88,70;CIAGAN=60,70,65,55,78,50"
        metrics: List of metric names to map values to.

    Returns:
        Dict of {method_name: {metric: value}}.
    """
    result = {}
    for method_entry in radar_str.split(';'):
        method_entry = method_entry.strip()
        if '=' not in method_entry:
            continue
        name, values_str = method_entry.split('=', 1)
        name = name.strip()
        values = [float(v.strip()) for v in values_str.split(',')]
        metric_dict = {}
        for i, metric in enumerate(metrics):
            if i < len(values):
                metric_dict[metric] = values[i]
        result[name] = metric_dict
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate visualizations for face de-identification analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Side-by-side comparison (horizontal layout with custom method names)
  python scripts/run_visualization.py --viz_type side_by_side \\
      --original_dir /path/to/original \\
      --deid_dirs "blur=/path/to/blur,ciagan=/path/to/ciagan" \\
      --method_names "Gaussian Blur" "CIAGAN" \\
      --num_images 5

  # Radar chart with manual values (reference-based normalization)
  python scripts/run_visualization.py --viz_type radar_chart \\
      --radar_values "k-Same=17.81,59.16,20.49,15.55,0.3973,32.56;DeID-rPPG=10.78,94.03,70.28,69.38,0.3627,0.39" \\
      --radar_reference "10.18,94.39,71.99,70.54,0.3601,0.39"

  # Radar chart from evaluation results
  python scripts/run_visualization.py --viz_type radar_chart \\
      --results_dirs /path/to/results1.json,/path/to/results2.json \\
      --method_names blur,ciagan
"""
    )

    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (CLI args override config values)')

    # Common arguments
    parser.add_argument('--viz_type', type=str, required=True,
                        choices=['side_by_side', 'attribute_overlay', 'embedding_tsne', 'radar_chart'],
                        help='Type of visualization to generate')
    parser.add_argument('--save_dir', type=str, default='runs/vis/',
                        help='Output directory for visualizations (default: runs/vis/)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for output images (default: 300)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for model inference (default: cuda)')

    # Image-based visualizers (side_by_side, attribute_overlay, embedding_tsne)
    parser.add_argument('--original_dir', type=str, default=None,
                        help='Path to original dataset images')
    parser.add_argument('--deid_dirs', type=str, default=None,
                        help='Comma-separated "name=path" pairs for deid outputs '
                             '(e.g. "blur=/path/blur,ciagan=/path/ciagan")')
    parser.add_argument('--num_images', type=int, default=None,
                        help='Number of sample images (default: 5 for grids, 200 for tsne)')

    # Side-by-side specific
    parser.add_argument('--show_bbox', action='store_true',
                        help='Draw face bounding boxes on images (side_by_side)')

    # Attribute overlay specific
    parser.add_argument('--attributes', type=str, default='age,gender,expression',
                        help='Comma-separated attributes to predict (attribute_overlay, '
                             'default: age,gender,expression)')

    # Embedding t-SNE specific
    parser.add_argument('--model_name', type=str, default='arcface',
                        choices=['arcface', 'cosface', 'adaface'],
                        help='Face recognition model for embeddings (embedding_tsne, default: arcface)')
    parser.add_argument('--perplexity', type=int, default=30,
                        help='t-SNE perplexity parameter (embedding_tsne, default: 30)')

    # Method display names (used by all visualization types)
    parser.add_argument('--method_names', type=str, nargs='+', default=None,
                        help='Display names for methods (space or comma-separated). '
                             'For image visualizers, overrides names from --deid_dirs. '
                             'For radar_chart, labels the methods in the legend.')

    # Radar chart specific
    parser.add_argument('--results_dirs', type=str, nargs='+', default=None,
                        help='Paths to results.json files (space or comma-separated)')
    parser.add_argument('--metrics', type=str, nargs='+', default=None,
                        help='Metric names to display (space or comma-separated, '
                             'default: Age Gender Ethnicity Expression Landmark rPPG)')
    parser.add_argument('--radar_values', type=str, default=None,
                        help='Manually set radar chart values. Format: '
                             '"Method1=v1,v2,...,v6;Method2=v1,v2,...,v6" '
                             'where values correspond to the metrics in order. '
                             'Example: "Blur=80,90,85,75,88,70;CIAGAN=60,70,65,55,78,50"')
    parser.add_argument('--radar_reference', type=str, default=None,
                        help='Reference baseline values (e.g., Original) for normalization. '
                             'Format: "v1,v2,...,v6" corresponding to metrics in order. '
                             'Each method is scored by how well it preserves the reference. '
                             'Example: "10.18,94.39,71.99,70.54,0.3601,0.39"')

    return load_config_into_args(parser)


def main():
    args = parse_args()

    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(args.save_dir) / f'{args.viz_type}_{timestamp}'
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Visualization Generation")
    print("=" * 70)
    print(f"Type: {args.viz_type}")
    print(f"Save directory: {save_dir}")
    print(f"Device: {args.device}")

    # Build config dict
    config = {
        'viz_type': args.viz_type,
        'save_dir': str(save_dir),
        'dpi': args.dpi,
        'device': args.device,
    }

    # Parse deid_dirs
    if args.deid_dirs:
        config['deid_dirs'] = parse_deid_dirs(args.deid_dirs)
        print(f"De-identification methods: {list(config['deid_dirs'].keys())}")

    if args.original_dir:
        config['original_dir'] = args.original_dir
        print(f"Original directory: {args.original_dir}")

    # Parse method display names (shared across all visualization types)
    if args.method_names:
        method_display_names = flatten_list_arg(args.method_names)
        config['method_display_names'] = method_display_names
        print(f"Method display names: {method_display_names}")

    # Type-specific config
    if args.viz_type == 'side_by_side':
        config['num_images'] = args.num_images or 5
        config['show_bbox'] = args.show_bbox
        print(f"Number of images: {config['num_images']}")
        if args.show_bbox:
            print("Bounding boxes: enabled")

    elif args.viz_type == 'attribute_overlay':
        config['num_images'] = args.num_images or 5
        config['attributes'] = [a.strip() for a in args.attributes.split(',')]
        print(f"Number of images: {config['num_images']}")
        print(f"Attributes: {config['attributes']}")

    elif args.viz_type == 'embedding_tsne':
        config['num_images'] = args.num_images or 200
        config['model_name'] = args.model_name
        config['perplexity'] = args.perplexity
        print(f"Number of images: {config['num_images']}")
        print(f"Recognition model: {args.model_name}")
        print(f"Perplexity: {args.perplexity}")

    elif args.viz_type == 'radar_chart':
        if args.results_dirs:
            config['results_dirs'] = flatten_list_arg(args.results_dirs)
            print(f"Results files: {config['results_dirs']}")
        if args.method_names:
            config['method_names'] = flatten_list_arg(args.method_names)
            print(f"Method names: {config['method_names']}")
        if args.metrics:
            config['metrics'] = flatten_list_arg(args.metrics)
            print(f"Metrics: {config['metrics']}")
        else:
            from core.visualization.radar_chart import DEFAULT_UTILITY_METRICS
            config['metrics'] = DEFAULT_UTILITY_METRICS
        if args.radar_values:
            radar_values = parse_radar_values(args.radar_values, config['metrics'])
            config['radar_values'] = radar_values
            print(f"Radar values (manual): {list(radar_values.keys())}")
        if args.radar_reference:
            ref_values = [float(v.strip()) for v in args.radar_reference.split(',')]
            ref_dict = {}
            for i, metric in enumerate(config['metrics']):
                if i < len(ref_values):
                    ref_dict[metric] = ref_values[i]
            config['radar_reference'] = ref_dict
            print(f"Radar reference: {ref_dict}")

    print("=" * 70)

    # Create and run visualizer
    visualizer = get_visualizer(config)
    output_path = visualizer.generate()

    print("=" * 70)
    print(f"Visualization saved to: {output_path}")
    print("Done!")


if __name__ == '__main__':
    main()
