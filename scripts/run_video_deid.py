#!/usr/bin/env python3
"""
Video face de-identification using naive methods.

This script:
1. Discovers video files or frame-sequence directories from input path
2. Detects faces using RetinaFace
3. Applies naive de-identification (blur, pixelate, or mask)
4. Writes de-identified frames/videos to output directory
5. Saves configuration and per-video statistics to config.yaml

Supports:
- Video files (.mp4, .avi, .mov, .mkv, .wmv, .flv, .webm)
- Frame-sequence directories (sorted image files)
- Real-time mode via --detect_every_n (skip detection on intermediate frames)

Usage:
    # Process PURE frame sequences with blur
    python scripts/run_video_deid.py \\
        --input /path/to/PURE --input_type frame_sequence \\
        --method blur --kernel_size 60 --save_dir output/

    # Process video files with pixelation (real-time mode)
    python scripts/run_video_deid.py \\
        --input /path/to/videos --input_type video \\
        --method pixelate --block_size 8 --detect_every_n 5 --save_dir output/

    # Auto-detect input type
    python scripts/run_video_deid.py \\
        --input /path/to/input --method mask --save_dir output/
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.identity.retinaface import FaceDetector
from core.fdeid.naive import (
    GaussianBlurDeIdentifier,
    PixelateDeIdentifier,
    MaskDeIdentifier
)
from core.video import VideoProcessor, discover_sources
from core.config_utils import load_config_into_args


# Model paths
RETINAFACE_MODEL = './weight/retinaface_pre_trained/Resnet50_Final.pth'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Video face de-identification using naive methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # PURE frame sequences with blur
  python scripts/run_video_deid.py --input /path/to/PURE --input_type frame_sequence --method blur --save_dir output/

  # Video files with pixelation (real-time)
  python scripts/run_video_deid.py --input /path/to/videos --method pixelate --detect_every_n 5 --save_dir output/

  # Auto-detect with mask
  python scripts/run_video_deid.py --input /path/to/input --method mask --save_dir output/
"""
    )

    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (CLI args override config values)')

    # Input arguments
    parser.add_argument('--input', type=str, required=True,
                        help='Path to video file, directory of videos, or frame-sequence directory')
    parser.add_argument('--input_type', type=str, default='auto',
                        choices=['auto', 'video', 'frame_sequence'],
                        help='Input type (default: auto)')

    # Method arguments
    parser.add_argument('--method', type=str, default='blur',
                        choices=['blur', 'pixelate', 'mask'],
                        help='De-identification method (default: blur)')
    parser.add_argument('--kernel_size', type=int, default=60,
                        help='Kernel size for blur method (default: 60)')
    parser.add_argument('--block_size', type=int, default=8,
                        help='Block size for pixelate method (default: 8)')
    parser.add_argument('--mask_color', type=int, nargs=3, default=[0, 0, 0],
                        help='Mask color BGR (default: 0 0 0)')

    # Video processing arguments
    parser.add_argument('--detect_every_n', type=int, default=1,
                        help='Run face detection every N frames (default: 1, set >1 for speed)')
    parser.add_argument('--fps', type=float, default=None,
                        help='Output FPS (default: match input)')
    parser.add_argument('--codec', type=str, default='mp4v',
                        help='FourCC codec for video output (default: mp4v)')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum frames per video (default: all)')
    parser.add_argument('--max_videos', type=int, default=None,
                        help='Maximum number of videos to process (default: all)')

    # Output arguments
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for face detection (default: cuda)')

    return load_config_into_args(parser)


def create_deidentifier(method: str, config: dict):
    """Create de-identifier based on method name."""
    if method == 'blur':
        return GaussianBlurDeIdentifier(config)
    elif method == 'pixelate':
        return PixelateDeIdentifier(config)
    elif method == 'mask':
        return MaskDeIdentifier(config)
    else:
        raise ValueError(f"Unknown method: {method}")


def main():
    args = parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Setup save directory
    save_dir = Path(args.save_dir)
    data_dir = save_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Video Face De-identification Pipeline")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Input type: {args.input_type}")
    print(f"Method: {args.method}")
    print(f"Detect every N frames: {args.detect_every_n}")
    print(f"Save directory: {save_dir}")

    # Method-specific parameters
    if args.method == 'blur':
        print(f"Kernel size: {args.kernel_size}")
    elif args.method == 'pixelate':
        print(f"Block size: {args.block_size}")
    elif args.method == 'mask':
        print(f"Mask color: {args.mask_color}")
    if args.max_frames:
        print(f"Max frames per video: {args.max_frames}")
    if args.max_videos:
        print(f"Max videos: {args.max_videos}")
    print("=" * 60)

    # Discover video sources
    print("\nDiscovering video sources...")
    sources = discover_sources(args.input, args.input_type)

    if not sources:
        print("Error: No video sources found!")
        return

    if args.max_videos:
        sources = sources[:args.max_videos]

    print(f"Found {len(sources)} video source(s)")
    for src_path, src_type in sources:
        print(f"  [{src_type}] {src_path}")

    # Initialize face detector
    print("\nInitializing face detector...")
    detector = FaceDetector(
        model_path=RETINAFACE_MODEL,
        network='resnet50',
        confidence_threshold=0.5,
        device=args.device
    )

    # Initialize de-identifier
    print(f"Initializing {args.method} de-identifier...")
    deid_config = {
        'kernel_size': args.kernel_size,
        'block_size': args.block_size,
        'mask_color': tuple(args.mask_color),
        'device': args.device
    }
    deidentifier = create_deidentifier(args.method, deid_config)

    # Initialize processor
    processor = VideoProcessor(
        detector=detector,
        deidentifier=deidentifier,
        detect_every_n=args.detect_every_n,
        verbose=True
    )

    # Process each video source
    print(f"\nProcessing {len(sources)} video(s)...")
    all_stats = []
    input_base = Path(args.input)

    for idx, (src_path, src_type) in enumerate(sources):
        print(f"\n[{idx + 1}/{len(sources)}] Processing: {src_path}")

        # Determine output path preserving directory structure
        src_p = Path(src_path)
        if src_p == input_base or src_p.parent == input_base.parent:
            # Single file or single frame dir
            rel_name = src_p.name
        else:
            try:
                rel_name = str(src_p.relative_to(input_base))
            except ValueError:
                rel_name = src_p.name

        if src_type == 'frame_sequence':
            output_path = str(data_dir / rel_name)
        else:
            output_path = str(data_dir / rel_name)

        stats = processor.process_video(
            input_path=src_path,
            output_path=output_path,
            max_frames=args.max_frames,
            fps=args.fps,
            codec=args.codec
        )
        all_stats.append(stats)

    # Summary
    total_frames = sum(s['frames_processed'] for s in all_stats)
    total_faces = sum(s['faces_detected'] for s in all_stats)
    total_time = sum(s['elapsed_seconds'] for s in all_stats)
    avg_fps = total_frames / total_time if total_time > 0 else 0

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Videos processed: {len(all_stats)}")
    print(f"Total frames: {total_frames}")
    print(f"Total faces detected: {total_faces}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average FPS: {avg_fps:.1f}")
    print("=" * 60)

    # Save configuration
    config = {
        'timestamp': timestamp,
        'pipeline': 'video',
        'method_type': 'naive',
        'method_name': args.method,
        'input_path': args.input,
        'input_type': args.input_type,
        'parameters': {
            'kernel_size': args.kernel_size,
            'block_size': args.block_size,
            'mask_color': args.mask_color,
        },
        'processing': {
            'detect_every_n': args.detect_every_n,
            'fps': args.fps,
            'codec': args.codec,
            'max_frames': args.max_frames,
            'max_videos': args.max_videos,
        },
        'statistics': {
            'videos_processed': len(all_stats),
            'total_frames': total_frames,
            'total_faces_detected': total_faces,
            'total_time_seconds': round(total_time, 2),
            'average_fps': round(avg_fps, 2),
        },
        'per_video_stats': all_stats,
        'device': args.device,
        'detector_model': RETINAFACE_MODEL,
    }

    config_file = save_dir / 'config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\nConfiguration saved to: {config_file}")
    print(f"Results saved to: {save_dir}")
    print("  - Data: data/")
    print("  - Config: config.yaml")
    print("\nDone!")


if __name__ == '__main__':
    main()
