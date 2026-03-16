"""
FactorizePhys rPPG Predictor

Provides a high-level interface for rPPG signal extraction using FactorizePhys.
Implements the official post-processing pipeline for accurate heart rate estimation.
"""

import os
import glob
import json
import numpy as np
import cv2
import torch
from typing import Dict, List, Tuple, Optional, Union
from scipy.signal import butter, filtfilt
from scipy.sparse import spdiags
import scipy.signal

from .model import FactorizePhys, get_factorizephys_config


# Default model paths
DEFAULT_MODEL_PATH = './weight/rppg_pre_trained/PURE_FactorizePhys_FSAM_Res.pth'
DEFAULT_RETINAFACE_PATH = './weight/retinaface_pre_trained/Resnet50_Final.pth'
DEFAULT_YOLO5FACE_PATH = './weight/rppg_pre_trained/Y5sF_WFRGB.pth'

# PURE dataset parameters
PURE_FS = 30  # Sampling frequency (frames per second)
PURE_CHUNK_LENGTH = 160  # Number of frames per chunk


# ============================================================================
# Post-Processing Functions (from official FactorizePhys)
# ============================================================================

def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _detrend(input_signal, lambda_value):
    """
    Detrend PPG signal using sparse matrix method.

    Args:
        input_signal: Input signal array
        lambda_value: Regularization parameter (default: 100)

    Returns:
        Detrended signal
    """
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    detrended_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return detrended_signal


def _calculate_fft_hr(ppg_signal, fs=30, low_pass=0.6, high_pass=3.3):
    """
    Calculate heart rate based on PPG using Fast Fourier transform (FFT).
    Uses scipy.signal.periodogram with next power of 2 padding.

    Args:
        ppg_signal: PPG signal array
        fs: Sampling frequency in Hz
        low_pass: Low frequency cutoff in Hz (default: 0.6 = 36 BPM)
        high_pass: High frequency cutoff in Hz (default: 3.3 = 198 BPM)

    Returns:
        Heart rate in BPM
    """
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr


def _bandpass_filter(signal_data, fs, low_pass=0.6, high_pass=3.3):
    """
    Apply bandpass filter to signal.

    Args:
        signal_data: Input signal
        fs: Sampling frequency
        low_pass: Low frequency cutoff
        high_pass: High frequency cutoff

    Returns:
        Filtered signal
    """
    [b, a] = butter(1, [low_pass / fs * 2, high_pass / fs * 2], btype='bandpass')
    return filtfilt(b, a, np.double(signal_data))


def post_process_signal(signal_data, fs=30, diff_flag=False, use_bandpass=True):
    """
    Post-process rPPG signal following official FactorizePhys pipeline.

    Args:
        signal_data: Raw signal from model
        fs: Sampling frequency
        diff_flag: If True, signal is first derivative (integrate it)
        use_bandpass: Apply bandpass filter

    Returns:
        Post-processed signal
    """
    if diff_flag:
        # If predictions are 1st derivative of PPG signal, integrate
        signal_data = _detrend(np.cumsum(signal_data), 100)
    else:
        signal_data = _detrend(signal_data, 100)

    if use_bandpass:
        signal_data = _bandpass_filter(signal_data, fs)

    return signal_data


def calculate_hr_from_signal(signal_data, fs=30, method='FFT'):
    """
    Calculate heart rate from processed signal.

    Args:
        signal_data: Post-processed signal
        fs: Sampling frequency
        method: 'FFT' or 'Peak'

    Returns:
        Heart rate in BPM
    """
    if method == 'FFT':
        return _calculate_fft_hr(signal_data, fs=fs)
    else:
        # Peak detection
        peaks, _ = scipy.signal.find_peaks(signal_data)
        if len(peaks) < 2:
            return 0.0
        return 60 / (np.mean(np.diff(peaks)) / fs)


# ============================================================================
# Face Detection with RetinaFace
# ============================================================================

class FaceCropper:
    """
    Face cropper using RetinaFace or YOLO5Face for rPPG preprocessing.
    """

    def __init__(
        self,
        backend: str = 'retinaface',
        model_path: str = None,
        device: str = 'cuda',
        large_box_coef: float = 1.5,
        detection_frequency: int = 30,
        use_median_box: bool = False,
    ):
        """
        Initialize face cropper.

        Args:
            backend: 'retinaface' or 'yolo5face'
            model_path: Path to detector model
            device: Device for inference
            large_box_coef: Coefficient to enlarge face box (default: 1.5)
            detection_frequency: Run detection every N frames
            use_median_box: Use median of all detected boxes
        """
        self.device = device
        self.backend = backend
        self.large_box_coef = large_box_coef
        self.detection_frequency = detection_frequency
        self.use_median_box = use_median_box
        self.detector = None
        
        if backend == 'retinaface':
            self.model_path = model_path or DEFAULT_RETINAFACE_PATH
        elif backend == 'yolo5face':
            self.model_path = model_path or DEFAULT_YOLO5FACE_PATH
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Lazy load detector
        self._detector_loaded = False

    def _load_detector(self):
        """Lazy load the face detector."""
        if not self._detector_loaded:
            if self.backend == 'retinaface':
                from core.identity.retinaface import FaceDetector
                self.detector = FaceDetector(
                    model_path=self.model_path,
                    network='resnet50',
                    confidence_threshold=0.5,
                    nms_threshold=0.4,
                    device=self.device
                )
            elif self.backend == 'yolo5face':
                from core.identity.yolo5face.yolo5face import YOLO5Face
                self.detector = YOLO5Face(
                    model_path=self.model_path,
                    device=self.device
                )
            self._detector_loaded = True

    def _enlarge_box(self, bbox, img_shape, coef=1.5):
        """
        Enlarge bounding box by coefficient, ensuring it is square first.

        Args:
            bbox: [x1, y1, x2, y2]
            img_shape: (H, W) or (H, W, C)
            coef: Enlargement coefficient

        Returns:
            Enlarged bbox [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # Enforce square box (use max dimension)
        square_size = max(w, h)
        
        # Apply enlargement coefficient
        new_size = square_size * coef

        new_w = new_size
        new_h = new_size

        x1 = int(max(0, cx - new_w / 2))
        y1 = int(max(0, cy - new_h / 2))
        # Ensure we don't go out of bounds, but try to keep the square shape if possible
        # Ideally we should center it.
        # But simple clamping is what likely happens or is safe enough
        x2 = int(min(img_shape[1], cx + new_w / 2))
        y2 = int(min(img_shape[0], cy + new_h / 2))

        return [x1, y1, x2, y2]

    def crop_faces(
        self,
        frames: np.ndarray,
        target_size: int = 72
    ) -> np.ndarray:
        """
        Crop faces from video frames.

        Args:
            frames: Video frames (T, H, W, 3) in RGB format
            target_size: Output size for cropped faces

        Returns:
            Cropped face frames (T, target_size, target_size, 3)
        """
        self._load_detector()

        T, H, W, C = frames.shape
        cropped_frames = []

        all_boxes = []
        detection_indices = list(range(0, T, self.detection_frequency))

        # Detect faces at specified intervals
        for idx in detection_indices:
            # Detectors usually expect BGR?
            # RetinaFace expects BGR in this toolbox.
            # YOLO5Face.detect_face does a deepcopy but doesn't explicitly convert.
            # However, YOLO models usually trained on RGB or BGR. 
            # Original YOLO5Face code:
            # img0 = cv2.imread(path) # BGR
            # So let's assume BGR.
            frame_bgr = cv2.cvtColor(frames[idx], cv2.COLOR_RGB2BGR)
            
            if self.backend == 'retinaface':
                detections = self.detector.detect(frame_bgr)
                if len(detections) > 0:
                    # Use largest face
                    largest = max(detections, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
                    bbox = self._enlarge_box(largest.bbox, (H, W), self.large_box_coef)
                    all_boxes.append(bbox)
            elif self.backend == 'yolo5face':
                 # YOLO5Face detect_face returns [x1, y1, x2, y2] for the largest/main face or None
                 res = self.detector.detect_face(frame_bgr)
                 if res is not None:
                     bbox = self._enlarge_box(res, (H, W), self.large_box_coef)
                     all_boxes.append(bbox)

        if len(all_boxes) == 0:
            # No face detected, return resized full frames
            print("Warning: No face detected, using full frame")
            return np.array([cv2.resize(f, (target_size, target_size), interpolation=cv2.INTER_AREA) for f in frames])

        # Use median box if enabled
        if self.use_median_box:
            median_box = np.median(all_boxes, axis=0).astype(int)
            boxes_per_frame = [median_box] * T
        else:
            # ORIGINAL IMPLEMENTATION LOGIC:
            # Use the box from the corresponding detection frame for all frames in that interval
            # i.e. frames [0, 30) use box from frame 0
            # frames [30, 60) use box from frame 30
            boxes_per_frame = []
            for i in range(T):
                 # Find which detection index covers this frame
                 # indices are [0, 30, 60, ...]
                 # For i=0..29, we want index 0 (which is at index 0 of all_boxes)
                 # For i=30..59, we want index 30 (which is at index 1 of all_boxes)
                 
                 box_idx = i // self.detection_frequency
                 
                 # Handle case where we might run out of boxes (e.g. if T is not multiple)
                 if box_idx >= len(all_boxes):
                     box_idx = len(all_boxes) - 1
                     
                 boxes_per_frame.append(all_boxes[box_idx])

        # Crop all frames
        for i, frame in enumerate(frames):
            x1, y1, x2, y2 = boxes_per_frame[i]
            x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(W, x2)), int(min(H, y2))

            if x2 > x1 and y2 > y1:
                face_crop = frame[y1:y2, x1:x2]
                face_crop = cv2.resize(face_crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
            else:
                face_crop = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_AREA)

            cropped_frames.append(face_crop)

        return np.array(cropped_frames)


# ============================================================================
# FactorizePhys Predictor
# ============================================================================

class FactorizePhysPredictor:
    """
    FactorizePhys-based rPPG signal predictor with official post-processing.
    """

    def __init__(
        self,
        model_path: str = None,
        device: str = 'cuda',
        frames: int = 160,
        image_size: int = 72,
        fs: float = 30.0,
        use_face_detection: bool = True,
        face_detector_backend: str = 'retinaface',
        retinaface_path: str = None,
        yolo5face_path: str = None,
        large_box_coef: float = 1.5,
    ):
        """
        Initialize FactorizePhys predictor.

        Args:
            model_path: Path to pretrained model weights
            device: Device to use for inference
            frames: Number of frames per chunk (default: 160)
            image_size: Input image size (default: 72)
            fs: Sampling frequency in Hz (default: 30.0)
            use_face_detection: Whether to use face detection (default: True)
            face_detector_backend: 'retinaface' or 'yolo5face'
            retinaface_path: Path to RetinaFace model
            yolo5face_path: Path to YOLO5Face model
            large_box_coef: Face box enlargement coefficient (default: 1.5)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.frames = frames
        self.image_size = image_size
        self.fs = fs
        self.use_face_detection = use_face_detection

        # Get config - IMPORTANT: Match original inference config!
        # The original uses MD_FSAM=False during inference (see PURE_PURE_FactorizePhys_FSAM_Res.yaml)
        # Model was trained with FSAM but inference uses base model without FSAM
        self.config = get_factorizephys_config(use_fsam=False, md_residual=True, md_inference=False)

        # Create model
        self.model = FactorizePhys(
            frames=frames,
            md_config=self.config,
            in_channels=3,
            device=self.device,
            debug=False
        )

        # Load weights
        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)

            # Remove 'module.' prefix if present (from DataParallel during training)
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]  # Remove 'module.' prefix
                else:
                    new_key = key
                new_state_dict[new_key] = value

            # Load with strict=False to allow missing FSAM keys (when MD_FSAM=False)
            missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys (expected when MD_FSAM=False): {len(missing_keys)} keys")
            if unexpected_keys:
                print(f"Unexpected keys (FSAM params when MD_FSAM=False): {len(unexpected_keys)} keys")
            print(f"FactorizePhys model loaded from {self.model_path}")
        else:
            print(f"Warning: Model weights not found at {self.model_path}")

        self.model.to(self.device)
        self.model.eval()

        # Initialize face cropper
        if use_face_detection:
            # Determine correct model path based on backend
            detector_model_path = None
            if face_detector_backend == 'retinaface':
                detector_model_path = retinaface_path
            elif face_detector_backend == 'yolo5face':
                detector_model_path = yolo5face_path

            self.face_cropper = FaceCropper(
                backend=face_detector_backend,
                model_path=detector_model_path,
                device=device,
                large_box_coef=large_box_coef,
                detection_frequency=30,
            )
        else:
            self.face_cropper = None

    def preprocess_frames(self, frames: np.ndarray, crop_face: bool = True) -> torch.Tensor:
        """
        Preprocess video frames for model input.

        Args:
            frames: Video frames (T, H, W, 3) in RGB format, uint8 or float
            crop_face: Whether to crop face region

        Returns:
            Preprocessed tensor (1, 3, T+1, H, W) - padded with 1 extra frame
        """
        # Face cropping
        if crop_face and self.face_cropper is not None:
            frames = self.face_cropper.crop_faces(frames, target_size=self.image_size)

        # IMPORTANT: The model expects data in 0-255 range as float32
        # The model internally applies torch.diff() and InstanceNorm3d
        # Do NOT normalize to 0-1 range!
        if frames.dtype == np.uint8:
            frames = frames.astype(np.float32)  # Keep 0-255 range

        # Resize if needed (after face crop, should already be correct size)
        if frames.shape[1] != self.image_size or frames.shape[2] != self.image_size:
            resized = []
            for frame in frames:
                resized.append(cv2.resize(frame, (self.image_size, self.image_size)))
            frames = np.array(resized)

        # Convert to tensor (T, H, W, C) -> (C, T, H, W)
        tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float()

        # Add batch dimension
        tensor = tensor.unsqueeze(0)  # (1, C, T, H, W)

        # CRITICAL: Pad with 1 extra frame (repeat the last frame)
        # This is required because torch.diff() in the model reduces temporal dim by 1
        # The original FactorizePhys trainer does this:
        #   last_frame = torch.unsqueeze(data[:, :, -1, :, :], 2).repeat(1, 1, 1, 1, 1)
        #   data = torch.cat((data, last_frame), 2)
        last_frame = tensor[:, :, -1:, :, :]  # (1, C, 1, H, W)
        tensor = torch.cat([tensor, last_frame], dim=2)  # (1, C, T+1, H, W)

        return tensor

    @torch.no_grad()
    def predict(self, frames: np.ndarray, crop_face: bool = True) -> Dict[str, any]:
        """
        Predict rPPG signal from video frames.

        Args:
            frames: Video frames (T, H, W, 3) in RGB format
                   T should be equal to self.frames (160 by default)
            crop_face: Whether to crop face region

        Returns:
            Dictionary containing:
            - rppg_raw: Raw rPPG signal from model
            - rppg: Post-processed rPPG signal
            - heart_rate: Estimated heart rate in BPM
        """
        # Preprocess
        tensor = self.preprocess_frames(frames, crop_face=crop_face)
        tensor = tensor.to(self.device)

        # Inference
        outputs = self.model(tensor)
        rppg_raw = outputs[0]  # First output is rPPG signal

        # Get raw output WITHOUT per-chunk normalization
        # Per-chunk normalization creates discontinuities at boundaries when concatenated
        # We'll normalize the full signal later in predict_video()
        rppg_raw_np = rppg_raw.cpu().numpy().flatten()

        # For single-chunk prediction (backward compatibility), normalize and process
        rppg_normalized = (rppg_raw_np - np.mean(rppg_raw_np)) / (np.std(rppg_raw_np) + 1e-8)
        rppg_processed = post_process_signal(
            rppg_normalized,
            fs=self.fs,
            diff_flag=False,
            use_bandpass=True
        )
        heart_rate = calculate_hr_from_signal(rppg_processed, fs=self.fs, method='FFT')

        return {
            'rppg_raw': rppg_raw_np,  # UN-normalized raw output for concatenation
            'rppg': rppg_processed,    # Processed output for single-chunk use
            'heart_rate': heart_rate,  # HR for single-chunk use
        }

    def predict_video(
        self,
        video_frames: np.ndarray,
        overlap: int = 0,
        crop_face: bool = True
    ) -> Dict[str, any]:
        """
        Predict rPPG signal from a full video by processing chunks.

        IMPORTANT: This method matches the original FactorizePhys evaluation pipeline:
        1. Process video in chunks, get RAW model outputs
        2. Concatenate RAW outputs into full video signal
        3. Apply post-processing (detrend + bandpass) to FULL signal
        4. Calculate HR from the FULL post-processed signal

        Args:
            video_frames: Video frames (T, H, W, 3) where T can be any length
            overlap: Number of overlapping frames between chunks (default: 0)
            crop_face: Whether to crop face region

        Returns:
            Dictionary containing:
            - rppg_raw: Concatenated raw rPPG signal
            - rppg: Post-processed rPPG signal (full video)
            - avg_heart_rate: Heart rate from full signal
        """
        T = video_frames.shape[0]
        chunk_size = self.frames
        step = chunk_size - overlap

        # Pre-crop all faces at once for consistency
        if crop_face and self.face_cropper is not None:
            video_frames = self.face_cropper.crop_faces(video_frames, target_size=self.image_size)
            crop_face = False  # Already cropped

        all_rppg_raw = []  # Collect RAW outputs, not post-processed

        start = 0
        while start + chunk_size <= T:
            chunk = video_frames[start:start + chunk_size]
            result = self.predict(chunk, crop_face=crop_face)
            all_rppg_raw.append(result['rppg_raw'])  # Use RAW output
            start += step

        # Handle remaining frames
        if start < T and T - start >= chunk_size // 2:
            chunk = video_frames[-chunk_size:]
            result = self.predict(chunk, crop_face=crop_face)
            if len(all_rppg_raw) > 0:
                overlap_len = (start + chunk_size) - T
                all_rppg_raw.append(result['rppg_raw'][overlap_len:])
            else:
                all_rppg_raw.append(result['rppg_raw'])

        # Concatenate RAW (un-normalized) signals
        rppg_raw = np.concatenate(all_rppg_raw) if all_rppg_raw else np.array([])

        # Apply post-processing to FULL signal (matches original FactorizePhys eval)
        # CRITICAL: Normalize the ENTIRE concatenated signal, not per-chunk
        # This avoids discontinuities at chunk boundaries that cause harmonic detection
        if len(rppg_raw) > 0:
            # Normalize full signal (z-score)
            rppg_normalized = (rppg_raw - np.mean(rppg_raw)) / (np.std(rppg_raw) + 1e-8)

            # Apply detrend + bandpass filter
            rppg_processed = post_process_signal(
                rppg_normalized,
                fs=self.fs,
                diff_flag=False,  # LABEL_TYPE=Standardized means no cumsum needed
                use_bandpass=True
            )
            # Calculate HR from FULL post-processed signal
            full_signal_hr = calculate_hr_from_signal(rppg_processed, fs=self.fs, method='FFT')
        else:
            rppg_processed = np.array([])
            full_signal_hr = 0.0

        return {
            'rppg_raw': rppg_raw,
            'rppg': rppg_processed,
            'avg_heart_rate': full_signal_hr,
        }


def create_factorizephys_predictor(
    model_path: str = None,
    device: str = 'cuda',
    frames: int = 160,
    image_size: int = 72,
    fs: float = 30.0,
    use_face_detection: bool = True,
    face_detector_backend: str = 'retinaface',
    retinaface_path: str = None,
    yolo5face_path: str = None,
) -> FactorizePhysPredictor:
    """
    Factory function to create FactorizePhys predictor.

    Args:
        model_path: Path to pretrained model weights
        device: Device to use for inference
        frames: Number of frames per chunk
        image_size: Input image size
        fs: Sampling frequency in Hz
        use_face_detection: Whether to use face detection
        face_detector_backend: 'retinaface' or 'yolo5face'
        retinaface_path: Path to RetinaFace model
        yolo5face_path: Path to YOLO5Face model

    Returns:
        FactorizePhysPredictor instance
    """
    return FactorizePhysPredictor(
        model_path=model_path,
        device=device,
        frames=frames,
        image_size=image_size,
        fs=fs,
        use_face_detection=use_face_detection,
        face_detector_backend=face_detector_backend,
        retinaface_path=retinaface_path,
        yolo5face_path=yolo5face_path,
    )


# ============================================================================
# PURE Dataset Utilities
# ============================================================================

def load_pure_video(video_dir: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load video frames from PURE dataset directory.

    Args:
        video_dir: Path to video directory containing PNG frames

    Returns:
        Tuple of (frames array, frame paths)
    """
    png_files = sorted(glob.glob(os.path.join(video_dir, 'Image*.png')))

    frames = []
    for png_path in png_files:
        img = cv2.imread(png_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)

    return np.array(frames), png_files


def load_pure_labels(json_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ground truth labels from PURE dataset JSON file.

    Args:
        json_path: Path to JSON label file

    Returns:
        Tuple of (waveform array, pulse_rate array)
    """
    with open(json_path, 'r') as f:
        labels = json.load(f)

    waveforms = []
    pulse_rates = []

    for entry in labels["/FullPackage"]:
        waveforms.append(entry["Value"]["waveform"])
        pulse_rates.append(entry["Value"]["pulseRate"])

    return np.array(waveforms), np.array(pulse_rates)


def calculate_hr_from_waveform(waveform: np.ndarray, fs: float = 30.0) -> float:
    """
    Calculate heart rate from ground truth waveform using official pipeline.

    Args:
        waveform: Ground truth waveform signal
        fs: Sampling frequency in Hz

    Returns:
        Heart rate in BPM
    """
    if len(waveform) == 0:
        return 0.0

    # Apply same post-processing as predictions
    waveform_processed = post_process_signal(waveform, fs=fs, diff_flag=False, use_bandpass=True)

    # Calculate HR using FFT
    return calculate_hr_from_signal(waveform_processed, fs=fs, method='FFT')


def compute_rppg_metrics(
    pred_hr: Union[float, List[float]],
    gt_hr: Union[float, List[float]],
) -> Dict[str, float]:
    """
    Compute rPPG evaluation metrics.

    Args:
        pred_hr: Predicted heart rate(s)
        gt_hr: Ground truth heart rate(s)

    Returns:
        Dictionary with metrics (MAE, RMSE, MAPE, correlation)
    """
    pred_hr = np.atleast_1d(pred_hr)
    gt_hr = np.atleast_1d(gt_hr)

    if len(pred_hr) != len(gt_hr):
        raise ValueError("Prediction and ground truth must have same length")

    if len(pred_hr) == 0:
        return {'mae': 0.0, 'rmse': 0.0, 'mape': 0.0, 'correlation': 0.0}

    # MAE
    mae = np.mean(np.abs(pred_hr - gt_hr))

    # RMSE
    rmse = np.sqrt(np.mean((pred_hr - gt_hr) ** 2))

    # MAPE
    mape = np.mean(np.abs((pred_hr - gt_hr) / (gt_hr + 1e-8))) * 100

    # Pearson correlation
    if len(pred_hr) > 1:
        correlation = np.corrcoef(pred_hr, gt_hr)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
    else:
        correlation = 1.0 if pred_hr[0] == gt_hr[0] else 0.0

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
        'correlation': float(correlation),
    }
