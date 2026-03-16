#!/usr/bin/env python3
"""
rPPG (remote Photoplethysmography) Extraction Module

Implements various rPPG extraction methods for heart rate estimation from facial videos:

Traditional Methods:
- POS (Plane-Orthogonal-to-Skin): Wang et al. 2017
- GREEN: Verkruysse et al. 2008
- CHROME: De Haan & Jeanne 2013

Deep Learning Methods:
- FactorizePhys: NeurIPS 2024

References:
- Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017).
  Algorithmic principles of remote PPG. IEEE TBME, 64(7), 1479-1491.
- Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008).
  Remote plethysmographic imaging using ambient light. Opt. Express, 16, 21434-21445.
- De Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based rPPG.
  IEEE TBME, 60(10), 2878-2886.
"""

import math
import warnings
from typing import Optional, Tuple, Union

import numpy as np
from scipy import signal, sparse


def detrend(input_signal: np.ndarray, lambda_value: float = 100) -> np.ndarray:
    """
    Detrend a signal using a regularization parameter.

    Args:
        input_signal: Input signal to detrend (N,) or (N, 1)
        lambda_value: Regularization parameter (default: 100)

    Returns:
        Detrended signal with same shape as input
    """
    signal_length = input_signal.shape[0]

    # Observation matrix
    H = np.identity(signal_length)

    # Second-order difference matrix
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = sparse.spdiags(
        diags_data, diags_index,
        (signal_length - 2), signal_length
    ).toarray()

    # Apply detrending
    filtered_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))),
        input_signal
    )

    return filtered_signal


def process_video(frames: np.ndarray) -> np.ndarray:
    """
    Calculate the spatial average RGB values for each frame.

    Args:
        frames: Video frames (T, H, W, 3) where T is number of frames

    Returns:
        RGB temporal signals (T, 3)
    """
    RGB = []
    for frame in frames:
        # Compute spatial average for each channel
        summation = np.sum(np.sum(frame, axis=0), axis=0)
        RGB.append(summation / (frame.shape[0] * frame.shape[1]))

    return np.asarray(RGB)


def calculate_heart_rate(
    bvp_signal: np.ndarray,
    fs: float,
    freq_min: float = 0.75,
    freq_max: float = 3.0
) -> Tuple[float, np.ndarray]:
    """
    Calculate heart rate from BVP signal using FFT.

    Args:
        bvp_signal: Blood volume pulse signal (T,)
        fs: Sampling frequency in Hz
        freq_min: Minimum heart rate frequency in Hz (default: 0.75 Hz = 45 BPM)
        freq_max: Maximum heart rate frequency in Hz (default: 3.0 Hz = 180 BPM)

    Returns:
        Tuple of (heart_rate_bpm, power_spectrum)
    """
    # Compute FFT
    fft_data = np.fft.rfft(bvp_signal)
    fft_freq = np.fft.rfftfreq(len(bvp_signal), 1.0 / fs)

    # Find peak in valid frequency range
    valid_idx = np.where((fft_freq >= freq_min) & (fft_freq <= freq_max))[0]

    if len(valid_idx) == 0:
        warnings.warn("No valid frequencies found in range")
        return 0.0, np.abs(fft_data)

    fft_power = np.abs(fft_data[valid_idx])
    peak_idx = valid_idx[np.argmax(fft_power)]

    # Convert to BPM
    heart_rate_hz = fft_freq[peak_idx]
    heart_rate_bpm = heart_rate_hz * 60.0

    return heart_rate_bpm, np.abs(fft_data)


class rPPGExtractor:
    """
    Base class for rPPG signal extraction from facial videos.
    """

    def __init__(self, method: str = 'pos', fs: float = 30.0):
        """
        Initialize rPPG extractor.

        Args:
            method: Extraction method ('pos', 'green', 'chrome')
            fs: Sampling frequency (frame rate) in Hz (default: 30.0)
        """
        self.method = method.lower()
        self.fs = fs

        if self.method not in ['pos', 'green', 'chrome']:
            raise ValueError(f"Unknown method: {method}. Choose from: pos, green, chrome")

    def extract_bvp(self, frames: np.ndarray) -> np.ndarray:
        """
        Extract BVP (Blood Volume Pulse) signal from video frames.

        Args:
            frames: Facial region frames (T, H, W, 3) in RGB format, range [0, 255]
                   or normalized [0, 1]

        Returns:
            BVP signal (T,)
        """
        # Ensure frames are in correct range
        if frames.max() > 1.0:
            frames = frames / 255.0

        if self.method == 'pos':
            return self._extract_pos(frames)
        elif self.method == 'green':
            return self._extract_green(frames)
        elif self.method == 'chrome':
            return self._extract_chrome(frames)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _extract_pos(self, frames: np.ndarray) -> np.ndarray:
        """
        POS (Plane-Orthogonal-to-Skin) method.

        Reference: Wang et al. 2017
        """
        WinSec = 1.6
        RGB = process_video(frames)
        N = RGB.shape[0]
        H = np.zeros((1, N))
        l = math.ceil(WinSec * self.fs)

        for n in range(N):
            m = n - l
            if m >= 0:
                # Normalize by temporal mean
                Cn = np.true_divide(RGB[m:n, :], np.mean(RGB[m:n, :], axis=0))
                Cn = np.mat(Cn).H

                # Projection onto plane orthogonal to skin tone
                S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)

                # Build pulse signal
                h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
                mean_h = np.mean(h)

                for temp in range(h.shape[1]):
                    h[0, temp] = h[0, temp] - mean_h

                H[0, m:n] = H[0, m:n] + (h[0])

        BVP = H

        # Detrend
        BVP = detrend(np.mat(BVP).H, 100)
        BVP = np.asarray(np.transpose(BVP))[0]

        # Bandpass filter (45-180 BPM)
        b, a = signal.butter(1, [0.75 / self.fs * 2, 3 / self.fs * 2], btype='bandpass')
        BVP = signal.filtfilt(b, a, BVP.astype(np.double))

        return BVP

    def _extract_green(self, frames: np.ndarray) -> np.ndarray:
        """
        GREEN channel method.

        Reference: Verkruysse et al. 2008
        """
        RGB = process_video(frames)
        # Extract green channel (index 1)
        BVP = RGB[:, 1]
        return BVP

    def _extract_chrome(self, frames: np.ndarray) -> np.ndarray:
        """
        CHROME (Chrominance-based) method.

        Reference: De Haan & Jeanne 2013
        """
        LPF = 0.7
        HPF = 2.5
        WinSec = 1.6

        RGB = process_video(frames)
        FN = RGB.shape[0]
        NyquistF = 1 / 2 * self.fs
        B, A = signal.butter(3, [LPF / NyquistF, HPF / NyquistF], 'bandpass')

        WinL = math.ceil(WinSec * self.fs)
        if (WinL % 2):
            WinL = WinL + 1

        NWin = math.floor((FN - WinL // 2) / (WinL // 2))
        WinS = 0
        WinM = int(WinS + WinL // 2)
        WinE = WinS + WinL
        totallen = (WinL // 2) * (NWin + 1)
        S = np.zeros(totallen)

        for i in range(NWin):
            RGBBase = np.mean(RGB[WinS:WinE, :], axis=0)
            RGBNorm = np.zeros((WinE - WinS, 3))

            for temp in range(WinS, WinE):
                RGBNorm[temp - WinS] = np.true_divide(RGB[temp], RGBBase)

            # Chrominance signals
            Xs = np.squeeze(3 * RGBNorm[:, 0] - 2 * RGBNorm[:, 1])
            Ys = np.squeeze(1.5 * RGBNorm[:, 0] + RGBNorm[:, 1] - 1.5 * RGBNorm[:, 2])

            # Bandpass filter
            Xf = signal.filtfilt(B, A, Xs, axis=0)
            Yf = signal.filtfilt(B, A, Ys)

            # Adaptive weighting
            Alpha = np.std(Xf) / np.std(Yf)
            SWin = Xf - Alpha * Yf
            SWin = np.multiply(SWin, signal.windows.hann(WinL))

            # Overlap-add
            S[WinS:WinM] = S[WinS:WinM] + SWin[:int(WinL // 2)]
            S[WinM:WinE] = SWin[int(WinL // 2):]

            WinS = WinM
            WinM = WinS + WinL // 2
            WinE = WinS + WinL

        BVP = S
        return BVP

    def estimate_heart_rate(
        self,
        frames: np.ndarray,
        freq_min: float = 0.75,
        freq_max: float = 3.0
    ) -> Tuple[float, np.ndarray]:
        """
        Estimate heart rate from video frames.

        Args:
            frames: Facial region frames (T, H, W, 3)
            freq_min: Minimum heart rate frequency in Hz (default: 0.75 Hz = 45 BPM)
            freq_max: Maximum heart rate frequency in Hz (default: 3.0 Hz = 180 BPM)

        Returns:
            Tuple of (heart_rate_bpm, bvp_signal)
        """
        bvp_signal = self.extract_bvp(frames)
        heart_rate, _ = calculate_heart_rate(bvp_signal, self.fs, freq_min, freq_max)
        return heart_rate, bvp_signal


def create_rppg_extractor(method: str = 'pos', fs: float = 30.0) -> rPPGExtractor:
    """
    Factory function to create an rPPG extractor.

    Args:
        method: Extraction method ('pos', 'green', 'chrome')
        fs: Sampling frequency in Hz

    Returns:
        rPPGExtractor instance
    """
    return rPPGExtractor(method=method, fs=fs)


# Import FactorizePhys components
from .factorizephys import (
    FactorizePhys,
    FactorizePhysPredictor,
    create_factorizephys_predictor,
    get_factorizephys_config,
    load_pure_video,
    load_pure_labels,
    calculate_hr_from_waveform,
    compute_rppg_metrics,
    DEFAULT_MODEL_PATH,
    PURE_FS,
    PURE_CHUNK_LENGTH,
)


__all__ = [
    # Traditional methods
    'rPPGExtractor',
    'create_rppg_extractor',
    'calculate_heart_rate',
    'detrend',
    'process_video',
    # FactorizePhys
    'FactorizePhys',
    'FactorizePhysPredictor',
    'create_factorizephys_predictor',
    'get_factorizephys_config',
    'load_pure_video',
    'load_pure_labels',
    'calculate_hr_from_waveform',
    'compute_rppg_metrics',
    'DEFAULT_MODEL_PATH',
    'PURE_FS',
    'PURE_CHUNK_LENGTH',
]
