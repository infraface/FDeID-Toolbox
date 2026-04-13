"""
rPPG (remote Photoplethysmography) Validator Module

Evaluates rPPG signal extraction quality and heart rate estimation accuracy.
Supports evaluation of rPPG preservation in de-identified facial videos.

Metrics:
- MAE (Mean Absolute Error): Average absolute difference in BPM
- RMSE (Root Mean Squared Error): Root mean squared difference in BPM
- Pearson Correlation: Correlation between BVP signals
- SNR (Signal-to-Noise Ratio): Quality of extracted BVP signal
"""

import os
import sys
import csv
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import signal
from scipy.stats import pearsonr

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.utility.rppg import rPPGExtractor, calculate_heart_rate
from .validator import BaseValidator


class rPPGDataset(Dataset):
    """
    Dataset for rPPG evaluation.

    Expects a CSV file with columns:
    - video_path: Path to facial video (NumPy array .npy file with shape (T, H, W, 3))
    - heart_rate: Ground truth heart rate in BPM

    Alternatively, can accept:
    - video_path, bvp_signal_path: Paths to video and ground truth BVP signal
    """

    def __init__(
        self,
        csv_path: str,
        root_dir: str = "",
        has_bvp: bool = False
    ):
        self.root_dir = root_dir
        self.has_bvp = has_bvp
        self.samples = []

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)

            # Check if header exists
            header = next(reader, None)
            if header and header[0].lower() in ['video_path', 'path', 'video']:
                # Header exists, skip it
                pass
            else:
                # No header, reset file pointer
                f.seek(0)
                reader = csv.reader(f)

            for row in reader:
                if has_bvp and len(row) >= 2:
                    # video_path, bvp_signal_path
                    self.samples.append((row[0], row[1], None))
                elif len(row) >= 2:
                    # video_path, heart_rate
                    self.samples.append((row[0], None, float(row[1])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.has_bvp:
            video_path, bvp_path, _ = self.samples[idx]
            full_video_path = os.path.join(self.root_dir, video_path)
            full_bvp_path = os.path.join(self.root_dir, bvp_path)

            try:
                # Load video frames
                frames = np.load(full_video_path)  # Expected shape: (T, H, W, 3)

                # Load ground truth BVP
                gt_bvp = np.load(full_bvp_path)  # Expected shape: (T,)

                return frames, gt_bvp, -1.0  # -1 indicates BVP mode
            except Exception as e:
                print(f"Error loading video/BVP {full_video_path}: {e}")
                # Return dummy data
                return np.zeros((100, 64, 64, 3)), np.zeros(100), -1.0
        else:
            video_path, _, heart_rate = self.samples[idx]
            full_video_path = os.path.join(self.root_dir, video_path)

            try:
                # Load video frames
                frames = np.load(full_video_path)  # Expected shape: (T, H, W, 3)
                # Return empty array instead of None to avoid DataLoader collation errors
                return frames, np.array([]), heart_rate
            except Exception as e:
                print(f"Error loading video {full_video_path}: {e}")
                # Return dummy data
                return np.zeros((100, 64, 64, 3)), np.array([]), 0.0


class rPPGValidator(BaseValidator):
    """
    Validator for rPPG extraction and heart rate estimation.

    Can evaluate:
    1. Heart rate estimation accuracy (if ground truth HR is provided)
    2. BVP signal quality (if ground truth BVP is provided)
    """

    def __init__(
        self,
        method: str = 'pos',
        fs: float = 30.0,
        device: str = 'cpu',  # rPPG methods run on CPU
        has_bvp_gt: bool = False
    ):
        """
        Initialize rPPGValidator.

        Args:
            method: rPPG extraction method ('pos', 'green', 'chrome')
            fs: Video frame rate in Hz (default: 30.0)
            device: Device (rPPG runs on CPU, so this is ignored)
            has_bvp_gt: Whether dataset has ground truth BVP signals
        """
        # Create rPPG extractor (not a torch model)
        self.extractor = rPPGExtractor(method=method, fs=fs)
        self.method = method
        self.fs = fs
        self.has_bvp_gt = has_bvp_gt

        # Note: We don't call super().__init__ because rPPG extractor is not a torch model
        self.device = 'cpu'
        self.results = []

    def get_dataloader(
        self,
        dataset_path: str,
        batch_size: int = 1,  # Process one video at a time
        num_workers: int = 0  # No multiprocessing for video data
    ) -> DataLoader:
        """
        Load dataset from CSV path.

        Note: batch_size should be 1 for rPPG as videos have different lengths
        """
        root_dir = os.path.dirname(dataset_path)

        dataset = rPPGDataset(
            csv_path=dataset_path,
            root_dir=root_dir,
            has_bvp=self.has_bvp_gt
        )

        return DataLoader(
            dataset,
            batch_size=1,  # Must be 1 for variable-length videos
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False
        )

    def preprocess(self, batch: Any) -> Tuple[Any, Any]:
        """
        Preprocess batch. Extract frames and targets.
        """
        frames, gt_bvp, heart_rate = batch

        # Remove batch dimension (batch_size is 1)
        frames = frames[0].numpy() if isinstance(frames, torch.Tensor) else frames[0]

        if gt_bvp is not None and len(gt_bvp[0]) > 0:
            gt_bvp = gt_bvp[0].numpy() if isinstance(gt_bvp, torch.Tensor) else gt_bvp[0]
        else:
            gt_bvp = None

        if isinstance(heart_rate, torch.Tensor):
            heart_rate = heart_rate.item()
        else:
            heart_rate = float(heart_rate[0]) if hasattr(heart_rate, '__len__') else float(heart_rate)

        return frames, (gt_bvp, heart_rate)

    def init_metrics(self) -> None:
        """Initialize metric accumulators."""
        self.results = []

    def update_metrics(self, outputs: Any, targets: Tuple[Any, float]) -> None:
        """
        Update metrics with predictions.

        Args:
            outputs: Tuple of (estimated_hr, bvp_signal)
            targets: Tuple of (gt_bvp or None, gt_heart_rate)
        """
        estimated_hr, bvp_signal = outputs
        gt_bvp, gt_heart_rate = targets

        self.results.append({
            'estimated_hr': estimated_hr,
            'bvp_signal': bvp_signal,
            'gt_bvp': gt_bvp,
            'gt_hr': gt_heart_rate
        })

    def finalize_metrics(self) -> Dict[str, float]:
        """
        Compute final metrics.

        Returns:
            Dictionary with MAE, RMSE, Pearson correlation (if BVP GT available)
        """
        if not self.results:
            return {}

        metrics = {}

        # Heart rate metrics (if GT HR is available)
        hr_errors = []
        for result in self.results:
            if result['gt_hr'] > 0:  # Valid GT heart rate
                error = abs(result['estimated_hr'] - result['gt_hr'])
                hr_errors.append(error)

        if hr_errors:
            hr_errors = np.array(hr_errors)
            metrics['hr_mae'] = float(np.mean(hr_errors))
            metrics['hr_rmse'] = float(np.sqrt(np.mean(hr_errors ** 2)))
            metrics['hr_std'] = float(np.std(hr_errors))

        # BVP signal metrics (if GT BVP is available)
        if self.has_bvp_gt:
            correlations = []
            snr_values = []

            for result in self.results:
                if result['gt_bvp'] is not None:
                    pred_bvp = result['bvp_signal']
                    gt_bvp = result['gt_bvp']

                    # Ensure same length (truncate to shorter)
                    min_len = min(len(pred_bvp), len(gt_bvp))
                    pred_bvp = pred_bvp[:min_len]
                    gt_bvp = gt_bvp[:min_len]

                    # Pearson correlation
                    if len(pred_bvp) > 1:
                        corr, _ = pearsonr(pred_bvp, gt_bvp)
                        if not np.isnan(corr):
                            correlations.append(corr)

                    # SNR of predicted signal
                    snr = self._calculate_snr(pred_bvp)
                    if not np.isnan(snr):
                        snr_values.append(snr)

            if correlations:
                metrics['bvp_correlation'] = float(np.mean(correlations))
                metrics['bvp_correlation_std'] = float(np.std(correlations))

            if snr_values:
                metrics['bvp_snr'] = float(np.mean(snr_values))
                metrics['bvp_snr_std'] = float(np.std(snr_values))

        return metrics

    def _calculate_snr(self, bvp_signal: np.ndarray) -> float:
        """
        Calculate Signal-to-Noise Ratio of BVP signal.

        Args:
            bvp_signal: BVP signal

        Returns:
            SNR in dB
        """
        # Compute power spectrum
        fft_data = np.fft.rfft(bvp_signal)
        fft_freq = np.fft.rfftfreq(len(bvp_signal), 1.0 / self.fs)
        power = np.abs(fft_data) ** 2

        # Signal power: in physiological range (0.75-3 Hz = 45-180 BPM)
        signal_idx = np.where((fft_freq >= 0.75) & (fft_freq <= 3.0))[0]
        if len(signal_idx) == 0:
            return np.nan

        signal_power = np.sum(power[signal_idx])

        # Noise power: outside physiological range
        noise_idx = np.where((fft_freq < 0.75) | (fft_freq > 3.0))[0]
        if len(noise_idx) == 0:
            return np.nan

        noise_power = np.sum(power[noise_idx])

        if noise_power == 0:
            return np.nan

        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    def __call__(
        self,
        dataloader: DataLoader,
        batch_size: int = 1,
        num_workers: int = 0
    ) -> Dict[str, float]:
        """
        Run rPPG validation loop.

        Args:
            dataloader: DataLoader for video dataset
            batch_size: Not used (always 1 for videos)
            num_workers: Not used

        Returns:
            Dictionary of metrics
        """
        from tqdm import tqdm

        self.init_metrics()

        pbar = tqdm(dataloader, desc=f"rPPG Validation ({self.method.upper()})")

        for batch in pbar:
            frames, targets = self.preprocess(batch)

            # Extract BVP and estimate heart rate
            try:
                estimated_hr, bvp_signal = self.extractor.estimate_heart_rate(frames)
                outputs = (estimated_hr, bvp_signal)
                self.update_metrics(outputs, targets)
            except Exception as e:
                print(f"Error processing video: {e}")
                continue

        return self.finalize_metrics()


def evaluate_rppg_preservation(
    original_videos_csv: str,
    deidentified_videos_csv: str,
    method: str = 'pos',
    fs: float = 30.0,
    has_bvp_gt: bool = False
) -> Dict[str, float]:
    """
    Evaluate rPPG preservation between original and de-identified videos.

    Args:
        original_videos_csv: CSV path for original videos
        deidentified_videos_csv: CSV path for de-identified videos
        method: rPPG extraction method
        fs: Frame rate in Hz
        has_bvp_gt: Whether ground truth BVP is available

    Returns:
        Dictionary with preservation metrics
    """
    # Validate original videos
    original_validator = rPPGValidator(
        method=method,
        fs=fs,
        has_bvp_gt=has_bvp_gt
    )

    # Validate de-identified videos
    deidentified_validator = rPPGValidator(
        method=method,
        fs=fs,
        has_bvp_gt=has_bvp_gt
    )

    # Load dataloaders
    original_dataloader = original_validator.get_dataloader(original_videos_csv)
    deidentified_dataloader = deidentified_validator.get_dataloader(deidentified_videos_csv)

    # Evaluate both
    print("Evaluating original videos...")
    original_metrics = original_validator(original_dataloader)

    print("Evaluating de-identified videos...")
    deidentified_metrics = deidentified_validator(deidentified_dataloader)

    # Compute preservation metrics
    preservation_metrics = {
        'original_hr_mae': original_metrics.get('hr_mae', 0.0),
        'deidentified_hr_mae': deidentified_metrics.get('hr_mae', 0.0),
        'hr_mae_degradation': deidentified_metrics.get('hr_mae', 0.0) - original_metrics.get('hr_mae', 0.0),
    }

    if has_bvp_gt:
        preservation_metrics.update({
            'original_bvp_correlation': original_metrics.get('bvp_correlation', 0.0),
            'deidentified_bvp_correlation': deidentified_metrics.get('bvp_correlation', 0.0),
            'correlation_degradation': original_metrics.get('bvp_correlation', 0.0) - deidentified_metrics.get('bvp_correlation', 0.0),
        })

    return preservation_metrics
