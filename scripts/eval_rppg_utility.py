#!/usr/bin/env python3
"""
rPPG Utility Evaluation Script

This script evaluates the utility preservation of de-identified face videos
by comparing rPPG (heart rate) predictions between original and de-identified
videos using the FactorizePhys model.

Implements the official FactorizePhys evaluation pipeline:
- Face detection using RetinaFace with 1.5x large box coefficient
- Signal detrending with lambda=100
- Bandpass filtering [0.6, 3.3] Hz
- FFT-based heart rate estimation with scipy.signal.periodogram

Metrics computed:
- Heart rate estimation accuracy (MAE, RMSE, MAPE)
- Correlation between original and de-identified predictions
- Comparison with ground truth (PURE dataset)

Usage:
    python scripts/eval_rppg_utility.py \
        --dataset_path /path/to/PURE \
        --deid_path /path/to/deid \
        --output_dir /path/to/output \
        --model_path /path/to/model.pth \
        --retinaface_path /path/to/retinaface.pth

Output:
    - results.json: Detailed per-video results
    - summary.txt: Summary statistics with method configuration
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np
import cv2

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config_utils import load_config_into_args
from core.utility.rppg.factorizephys import (
    create_factorizephys_predictor,
    load_pure_video,
    load_pure_labels,
    calculate_hr_from_waveform,
    compute_rppg_metrics,
    post_process_signal,
    calculate_hr_from_signal,
    PURE_FS,
    PURE_CHUNK_LENGTH,
    DEFAULT_MODEL_PATH,
    DEFAULT_RETINAFACE_PATH,
    DEFAULT_YOLO5FACE_PATH,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate rPPG utility preservation using FactorizePhys',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (CLI args override config values)')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to PURE dataset directory')
    parser.add_argument('--deid_path', type=str, default=None,
                        help='Path to de-identified videos (optional, if evaluating deid)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
                        help='Path to FactorizePhys pretrained model')
    parser.add_argument('--face_detector_backend', type=str, default='retinaface', choices=['retinaface', 'yolo5face'],
                        help='Face detector backend (retinaface or yolo5face)')
    parser.add_argument('--retinaface_path', type=str, default=DEFAULT_RETINAFACE_PATH,
                        help='Path to RetinaFace model for face detection')
    parser.add_argument('--yolo5face_path', type=str, default=DEFAULT_YOLO5FACE_PATH,
                        help='Path to YOLO5Face model for face detection')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--max_videos', type=int, default=None,
                        help='Maximum number of videos to process (default: all)')
    parser.add_argument('--image_size', type=int, default=72,
                        help='Input image size for model (default: 72)')
    parser.add_argument('--large_box_coef', type=float, default=1.5,
                        help='Face box enlargement coefficient (default: 1.5)')
    parser.add_argument('--use_face_detection', action='store_true', default=True,
                        help='Use face detection (default: True)')
    parser.add_argument('--no_face_detection', action='store_false', dest='use_face_detection',
                        help='Disable face detection')

    return load_config_into_args(parser)


def get_pure_subjects(dataset_path: Path):
    """
    Get list of PURE subjects (video directories).

    PURE dataset structure:
    PURE/
    |-- 01-01/
    |   |-- 01-01/
    |   |   |-- Image00001.png
    |   |   |-- Image00002.png
    |   |   |-- ...
    |-- 01-01.json
    |-- 01-02/
    |   |-- 01-02/
    |   |   |-- Image00001.png
    |   |   |-- ...
    |-- 01-02.json
    |-- ...

    Returns:
        List of (subject_id, video_dir, label_path) tuples
    """
    subjects = []

    for item in sorted(dataset_path.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            subject_id = item.name
            # Video frames are in a subdirectory with the same name
            video_dir = item / subject_id
            if not video_dir.exists():
                video_dir = item  # Try the main directory

            label_path = dataset_path / f"{subject_id}.json"

            if label_path.exists() and video_dir.exists():
                subjects.append((subject_id, video_dir, label_path))

    return subjects


def load_config(deid_path: Path):
    """Load configuration from de-identified directory."""
    if deid_path is None:
        return {}
    config_path = deid_path / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def resample_signal(signal, target_length):
    """Resample signal to target length."""
    if len(signal) == target_length:
        return signal
    from scipy import interpolate
    x = np.linspace(0, 1, len(signal))
    f = interpolate.interp1d(x, signal, kind='linear')
    x_new = np.linspace(0, 1, target_length)
    return f(x_new)


def main():
    args = parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Setup directories
    dataset_path = Path(args.dataset_path)
    deid_path = Path(args.deid_path) if args.deid_path else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("rPPG Utility Evaluation (FactorizePhys)")
    print("=" * 70)
    print(f"Dataset path: {dataset_path}")
    print(f"De-identified path: {deid_path}")
    print(f"Model path: {args.model_path}")
    print(f"Face Detector: {args.face_detector_backend}")
    if args.face_detector_backend == 'retinaface':
        print(f"Detector path: {args.retinaface_path}")
    else:
        print(f"Detector path: {args.yolo5face_path}")
    print(f"Output directory: {output_dir}")
    print(f"Use face detection: {args.use_face_detection}")
    print(f"Large box coefficient: {args.large_box_coef}")
    print("=" * 70)

    # Load method config if available
    method_config = load_config(deid_path)

    # Get PURE subjects
    print("\nFinding PURE subjects...")
    subjects = get_pure_subjects(dataset_path)

    if args.max_videos:
        subjects = subjects[:args.max_videos]

    print(f"Found {len(subjects)} subjects")

    if len(subjects) == 0:
        print("Error: No subjects found!")
        return

    # Initialize FactorizePhys predictor with face detection
    print("\nInitializing FactorizePhys predictor...")
    predictor = create_factorizephys_predictor(
        model_path=args.model_path,
        device=args.device,
        frames=PURE_CHUNK_LENGTH,
        image_size=args.image_size,
        fs=PURE_FS,
        use_face_detection=args.use_face_detection,
        face_detector_backend=args.face_detector_backend,
        retinaface_path=args.retinaface_path,
        yolo5face_path=args.yolo5face_path,
    )

    # Process videos
    print(f"\nProcessing {len(subjects)} videos...")

    results = []

    # Metrics accumulators
    orig_pred_hrs = []
    deid_pred_hrs = []
    gt_hrs = []

    for subject_id, video_dir, label_path in tqdm(subjects, desc="Evaluating"):
        try:
            # Load original video
            orig_frames, _ = load_pure_video(str(video_dir))

            if len(orig_frames) < PURE_CHUNK_LENGTH:
                print(f"Warning: Skipping {subject_id} - not enough frames ({len(orig_frames)})")
                continue

            # Load ground truth waveform and calculate HR using official pipeline
            waveform, pulse_rates = load_pure_labels(str(label_path))

            # Resample waveform to match video length
            waveform_resampled = resample_signal(waveform, len(orig_frames))

            # Calculate GT HR from waveform using the same post-processing pipeline
            gt_hr = calculate_hr_from_waveform(waveform_resampled, fs=PURE_FS)

            # Predict on original video (face detection is handled internally)
            orig_result = predictor.predict_video(orig_frames, overlap=0, crop_face=args.use_face_detection)
            orig_pred_hr = orig_result['avg_heart_rate']

            result_entry = {
                'subject_id': subject_id,
                'num_frames': len(orig_frames),
                'gt_heart_rate': float(gt_hr),
                'original': {
                    'pred_heart_rate': float(orig_pred_hr),
                    'hr_error': float(abs(orig_pred_hr - gt_hr)),
                },
            }

            orig_pred_hrs.append(orig_pred_hr)
            gt_hrs.append(gt_hr)

            # Process de-identified if available
            if deid_path is not None:
                deid_video_dir = deid_path / 'data' / subject_id / subject_id
                if not deid_video_dir.exists():
                    deid_video_dir = deid_path / 'data' / subject_id
                if not deid_video_dir.exists():
                    deid_video_dir = deid_path / subject_id / subject_id
                if not deid_video_dir.exists():
                    deid_video_dir = deid_path / subject_id

                if deid_video_dir.exists():
                    deid_frames, _ = load_pure_video(str(deid_video_dir))

                    if len(deid_frames) >= PURE_CHUNK_LENGTH:
                        deid_result = predictor.predict_video(deid_frames, overlap=0, crop_face=args.use_face_detection)
                        deid_pred_hr = deid_result['avg_heart_rate']

                        result_entry['deid'] = {
                            'pred_heart_rate': float(deid_pred_hr),
                            'hr_error': float(abs(deid_pred_hr - gt_hr)),
                        }
                        result_entry['consistency'] = {
                            'hr_diff': float(abs(orig_pred_hr - deid_pred_hr)),
                        }

                        deid_pred_hrs.append(deid_pred_hr)

            results.append(result_entry)

        except Exception as e:
            print(f"Warning: Error processing {subject_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(results) == 0:
        print("Error: No videos processed successfully!")
        return

    # Compute aggregate metrics
    print("\nComputing aggregate metrics...")

    # Original vs Ground Truth
    orig_metrics = compute_rppg_metrics(orig_pred_hrs, gt_hrs)

    aggregate_metrics = {
        'original_performance': {
            'mae': orig_metrics['mae'],
            'rmse': orig_metrics['rmse'],
            'mape': orig_metrics['mape'],
            'correlation': orig_metrics['correlation'],
        },
        'num_videos': len(results),
    }

    # De-identified metrics if available
    if len(deid_pred_hrs) > 0:
        # Deid vs Ground Truth
        deid_metrics = compute_rppg_metrics(deid_pred_hrs, gt_hrs[:len(deid_pred_hrs)])

        # Consistency (Original vs Deid)
        consistency_metrics = compute_rppg_metrics(deid_pred_hrs, orig_pred_hrs[:len(deid_pred_hrs)])

        aggregate_metrics['deid_performance'] = {
            'mae': deid_metrics['mae'],
            'rmse': deid_metrics['rmse'],
            'mape': deid_metrics['mape'],
            'correlation': deid_metrics['correlation'],
        }

        aggregate_metrics['consistency'] = {
            'mae': consistency_metrics['mae'],
            'rmse': consistency_metrics['rmse'],
            'correlation': consistency_metrics['correlation'],
        }

        aggregate_metrics['performance_drop'] = {
            'mae_increase': deid_metrics['mae'] - orig_metrics['mae'],
            'rmse_increase': deid_metrics['rmse'] - orig_metrics['rmse'],
        }

    # Save results
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'dataset_path': str(dataset_path),
            'deid_path': str(deid_path) if deid_path else None,
            'model_path': args.model_path,
            'retinaface_path': args.retinaface_path,
            'use_face_detection': args.use_face_detection,
            'large_box_coef': args.large_box_coef,
            'method_config': method_config,
            'aggregate_metrics': aggregate_metrics,
            'per_video_results': results,
        }, f, indent=2)
    print(f"Results saved to: {results_path}")

    # Generate summary
    summary_path = output_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("rPPG Utility Evaluation Summary\n")
        f.write("=" * 70 + "\n\n")

        f.write("Dataset Information:\n")
        f.write(f"  Dataset path: {dataset_path}\n")
        f.write(f"  De-identified path: {deid_path}\n")
        f.write(f"  Number of videos evaluated: {len(results)}\n")
        f.write(f"  Model: FactorizePhys (PURE_FSAM_Res)\n")
        f.write(f"  Face detection: {args.use_face_detection}\n")
        f.write(f"  Large box coefficient: {args.large_box_coef}\n\n")

        if method_config:
            f.write("De-identification Method Configuration:\n")
            f.write(f"  Method type: {method_config.get('method_type', 'N/A')}\n")
            f.write(f"  Method name: {method_config.get('method_name', 'N/A')}\n")
            if 'parameters' in method_config:
                f.write("  Parameters:\n")
                for k, v in method_config['parameters'].items():
                    f.write(f"    {k}: {v}\n")
            f.write("\n")

        f.write("-" * 70 + "\n")
        f.write("Original Performance (vs Ground Truth)\n")
        f.write("-" * 70 + "\n")
        f.write(f"  MAE:         {aggregate_metrics['original_performance']['mae']:.2f} BPM\n")
        f.write(f"  RMSE:        {aggregate_metrics['original_performance']['rmse']:.2f} BPM\n")
        f.write(f"  MAPE:        {aggregate_metrics['original_performance']['mape']:.2f}%\n")
        f.write(f"  Correlation: {aggregate_metrics['original_performance']['correlation']:.4f}\n")
        f.write("-" * 70 + "\n\n")

        if 'deid_performance' in aggregate_metrics:
            f.write("-" * 70 + "\n")
            f.write("De-identified Performance (vs Ground Truth)\n")
            f.write("-" * 70 + "\n")
            f.write(f"  MAE:         {aggregate_metrics['deid_performance']['mae']:.2f} BPM\n")
            f.write(f"  RMSE:        {aggregate_metrics['deid_performance']['rmse']:.2f} BPM\n")
            f.write(f"  MAPE:        {aggregate_metrics['deid_performance']['mape']:.2f}%\n")
            f.write(f"  Correlation: {aggregate_metrics['deid_performance']['correlation']:.4f}\n")
            f.write("-" * 70 + "\n\n")

            f.write("-" * 70 + "\n")
            f.write("Consistency (Original vs De-identified)\n")
            f.write("-" * 70 + "\n")
            f.write(f"  MAE:         {aggregate_metrics['consistency']['mae']:.2f} BPM\n")
            f.write(f"  RMSE:        {aggregate_metrics['consistency']['rmse']:.2f} BPM\n")
            f.write(f"  Correlation: {aggregate_metrics['consistency']['correlation']:.4f}\n")
            f.write("-" * 70 + "\n\n")

            f.write("-" * 70 + "\n")
            f.write("Performance Drop\n")
            f.write("-" * 70 + "\n")
            f.write(f"  MAE Increase:  {aggregate_metrics['performance_drop']['mae_increase']:.2f} BPM\n")
            f.write(f"  RMSE Increase: {aggregate_metrics['performance_drop']['rmse_increase']:.2f} BPM\n")
            f.write("-" * 70 + "\n\n")

        f.write("Notes:\n")
        f.write("  - MAE: Mean Absolute Error in BPM (lower is better)\n")
        f.write("  - RMSE: Root Mean Squared Error in BPM (lower is better)\n")
        f.write("  - MAPE: Mean Absolute Percentage Error (lower is better)\n")
        f.write("  - Correlation: Pearson correlation coefficient (higher is better)\n")
        f.write("  - Higher consistency indicates better utility preservation.\n")
        f.write("  - Paper reported: MAE=0.48, RMSE=1.39, MAPE=0.72%, Corr=0.998\n\n")

        f.write(f"Evaluation timestamp: {timestamp}\n")
        f.write("=" * 70 + "\n")

    print(f"Summary saved to: {summary_path}")

    # Print summary to console
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\nOriginal Performance (vs Ground Truth):")
    print("-" * 50)
    print(f"  MAE:         {aggregate_metrics['original_performance']['mae']:.2f} BPM")
    print(f"  RMSE:        {aggregate_metrics['original_performance']['rmse']:.2f} BPM")
    print(f"  MAPE:        {aggregate_metrics['original_performance']['mape']:.2f}%")
    print(f"  Correlation: {aggregate_metrics['original_performance']['correlation']:.4f}")
    print("-" * 50)
    print("  Paper:       MAE=0.48, RMSE=1.39, MAPE=0.72%, Corr=0.998")
    print("-" * 50)

    if 'deid_performance' in aggregate_metrics:
        print("\nDe-identified Performance (vs Ground Truth):")
        print("-" * 50)
        print(f"  MAE:         {aggregate_metrics['deid_performance']['mae']:.2f} BPM")
        print(f"  RMSE:        {aggregate_metrics['deid_performance']['rmse']:.2f} BPM")
        print("-" * 50)

        print("\nConsistency (Original vs De-identified):")
        print("-" * 50)
        print(f"  MAE:         {aggregate_metrics['consistency']['mae']:.2f} BPM")
        print(f"  Correlation: {aggregate_metrics['consistency']['correlation']:.4f}")
        print("-" * 50)

    print(f"\n  Videos evaluated: {len(results)}")
    print("=" * 70)
    print("\nDone!")


if __name__ == '__main__':
    main()
