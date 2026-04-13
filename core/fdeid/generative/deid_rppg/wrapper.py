"""
DeID-rPPG: Rethinking the tradeoff between utility and privacy in video-based remote PPG.

Wrapper for the DeID-rPPG implementation integrated into the toolbox.
This is a video-based de-identification method that preserves rPPG signals.

Note: This method is designed for video de-identification (64-frame chunks).
For single-frame processing, frames are replicated to form a pseudo-video.
"""

import os
import sys
import cv2
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple

# Ensure local imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from .AutoEncoder_3d import AutoEncoder
from ...base import BaseDeIdentifier

# Default paths
DEFAULT_AUTOENCODER_WEIGHTS = './weight/deid_rppg_pre_trained/autoencoder.pt'


class DeIDrPPGDeIdentifier(BaseDeIdentifier):
    """
    DeID-rPPG: Video-based face de-identification with rPPG preservation.

    This method uses a 3D AutoEncoder to de-identify faces in video frames
    while preserving the remote photoplethysmography (rPPG) signal for
    physiological measurements.

    Key Features:
    - Processes video in 64-frame chunks
    - Preserves rPPG signal for heart rate measurement
    - Uses 3D convolutions for temporal consistency
    - Input/output in range [-1, 1]

    Note:
    - For single images, frames are replicated to form a pseudo-video
    - Best results are achieved with actual video sequences
    - Model expects 128x128 face crops
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Configuration
        self.model_path = config.get('weights_path') or config.get('model_path') or DEFAULT_AUTOENCODER_WEIGHTS
        self.chunk_size = config.get('chunk_size', 64)  # Number of frames per chunk
        self.face_size = config.get('face_size', 128)  # Size of face crop (128x128)
        self.overlap = config.get('overlap', 0)  # Overlap between chunks (for video)

        # Initialize AutoEncoder
        self.model = AutoEncoder(frames=self.chunk_size)

        # Load weights - REQUIRED for meaningful results
        self._weights_loaded = False
        if self.model_path and os.path.exists(self.model_path):
            try:
                state_dict = torch.load(self.model_path, map_location='cpu', weights_only=True)
                # Handle DataParallel weights
                if any(k.startswith('module.') for k in state_dict.keys()):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                self.model.load_state_dict(state_dict)
                print(f"DeID-rPPG weights loaded from {self.model_path}")
                self._weights_loaded = True
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load DeID-rPPG weights from {self.model_path}: {e}\n"
                    f"DeID-rPPG requires trained weights. Please train the model first using:\n"
                    f"  sbatch jobs/train_deid_rppg.sh"
                )
        else:
            raise FileNotFoundError(
                f"DeID-rPPG weights not found at '{self.model_path}'.\n"
                f"DeID-rPPG requires trained weights to produce meaningful results.\n"
                f"Please either:\n"
                f"  1. Train the model: sbatch jobs/train_deid_rppg.sh\n"
                f"  2. Provide weights via config: {{'weights_path': '/path/to/autoencoder.pt'}}"
            )

        self.model.to(self.device).eval()

        # Frame buffer for video processing
        self._frame_buffer = []

    def _preprocess_frame(self, frame: np.ndarray, face_bbox: Optional[Tuple] = None) -> Tuple[np.ndarray, Tuple]:
        """
        Preprocess a single frame for the model.

        Args:
            frame: Input frame (H, W, C) in BGR format, range [0, 255]
            face_bbox: Optional face bounding box (x1, y1, x2, y2)

        Returns:
            face_crop: Cropped and resized face (face_size, face_size, 3)
            crop_info: (x1, y1, x2, y2, orig_h, orig_w) for restoring
        """
        h, w = frame.shape[:2]

        if face_bbox is not None:
            x1, y1, x2, y2 = map(int, face_bbox)
        else:
            # Use center crop if no bbox provided
            crop_size = min(h, w)
            x1 = (w - crop_size) // 2
            y1 = (h - crop_size) // 2
            x2 = x1 + crop_size
            y2 = y1 + crop_size

        # Ensure valid coordinates
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # Crop face
        face_crop = frame[y1:y2, x1:x2].copy()
        orig_h, orig_w = face_crop.shape[:2]

        # Resize to model input size
        if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
            face_crop = cv2.resize(face_crop, (self.face_size, self.face_size), interpolation=cv2.INTER_LINEAR)
        else:
            face_crop = np.zeros((self.face_size, self.face_size, 3), dtype=np.uint8)

        return face_crop, (x1, y1, x2, y2, orig_h, orig_w)

    def _postprocess_frame(self, processed_face: np.ndarray, original_frame: np.ndarray, crop_info: Tuple) -> np.ndarray:
        """
        Postprocess and paste the processed face back to the original frame.

        Args:
            processed_face: Processed face (face_size, face_size, 3)
            original_frame: Original frame to paste into
            crop_info: (x1, y1, x2, y2, orig_h, orig_w)

        Returns:
            Frame with processed face pasted back
        """
        x1, y1, x2, y2, orig_h, orig_w = crop_info
        result_frame = original_frame.copy()

        if orig_h > 0 and orig_w > 0:
            # Resize processed face back to original crop size
            restored_face = cv2.resize(processed_face, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            result_frame[y1:y2, x1:x2] = restored_face

        return result_frame

    def process_frame(self, frame: np.ndarray, face_bbox: Optional[Union[List, Tuple, np.ndarray]] = None, **kwargs) -> np.ndarray:
        """
        Apply de-identification to a single frame.

        Note: Since DeID-rPPG is designed for video, this method replicates
        the single frame to create a pseudo-video chunk for processing.

        Args:
            frame: Input image (H, W, C) in BGR format, range [0, 255]
            face_bbox: Optional face bounding box (x1, y1, x2, y2)
            **kwargs: Additional parameters

        Returns:
            De-identified frame
        """
        # Preprocess
        face_crop, crop_info = self._preprocess_frame(frame, face_bbox)

        # Convert BGR to RGB and normalize to [-1, 1]
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_tensor = torch.from_numpy(face_rgb).float().permute(2, 0, 1) / 255.0  # (3, H, W) in [0, 1]
        face_tensor = face_tensor * 2.0 - 1.0  # Normalize to [-1, 1]

        # Replicate to create pseudo-video: (3, T, H, W)
        face_video = face_tensor.unsqueeze(1).repeat(1, self.chunk_size, 1, 1)  # (3, T, H, W)
        face_video = face_video.unsqueeze(0).to(self.device)  # (1, 3, T, H, W)

        # Process through model
        with torch.no_grad():
            output_video = self.model(face_video)  # (1, 3, T, H, W) in [-1, 1]

        # Take the middle frame (most stable due to temporal context)
        middle_idx = self.chunk_size // 2
        output_frame = output_video[0, :, middle_idx, :, :]  # (3, H, W)

        # Convert back to numpy and BGR
        output_frame = (output_frame + 1.0) / 2.0  # Back to [0, 1]
        output_frame = output_frame.clamp(0, 1)
        output_np = (output_frame.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # (H, W, 3) RGB
        output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)

        # Postprocess - paste back to original frame
        result = self._postprocess_frame(output_bgr, frame, crop_info)

        return result

    def process_video_chunk(self, frames: List[np.ndarray], face_bboxes: Optional[List] = None, **kwargs) -> List[np.ndarray]:
        """
        Apply de-identification to a video chunk.

        This is the recommended method for video processing as it properly
        utilizes the temporal context of the 3D AutoEncoder.

        Args:
            frames: List of frames (each H, W, C in BGR format)
            face_bboxes: Optional list of face bounding boxes per frame
            **kwargs: Additional parameters

        Returns:
            List of de-identified frames
        """
        if len(frames) == 0:
            return []

        # Use first frame's bbox for all if single bbox provided
        if face_bboxes is None:
            face_bboxes = [None] * len(frames)
        elif len(face_bboxes) == 1 and len(frames) > 1:
            face_bboxes = face_bboxes * len(frames)

        # Preprocess all frames
        face_crops = []
        crop_infos = []
        for i, (frame, bbox) in enumerate(zip(frames, face_bboxes)):
            face_crop, crop_info = self._preprocess_frame(frame, bbox)
            face_crops.append(face_crop)
            crop_infos.append(crop_info)

        # Pad to chunk_size if needed
        num_frames = len(face_crops)
        if num_frames < self.chunk_size:
            # Pad by repeating last frame
            padding = [face_crops[-1]] * (self.chunk_size - num_frames)
            face_crops = face_crops + padding

        # Convert to tensor: (T, H, W, C) -> (C, T, H, W)
        face_rgb_list = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in face_crops[:self.chunk_size]]
        face_array = np.stack(face_rgb_list, axis=0)  # (T, H, W, 3)
        face_tensor = torch.from_numpy(face_array).float().permute(3, 0, 1, 2) / 255.0  # (3, T, H, W)
        face_tensor = face_tensor * 2.0 - 1.0  # Normalize to [-1, 1]
        face_video = face_tensor.unsqueeze(0).to(self.device)  # (1, 3, T, H, W)

        # Process through model
        with torch.no_grad():
            output_video = self.model(face_video)  # (1, 3, T, H, W) in [-1, 1]

        # Convert output back to numpy frames
        output_video = (output_video + 1.0) / 2.0  # Back to [0, 1]
        output_video = output_video.clamp(0, 1)
        output_video = output_video[0].cpu().permute(1, 2, 3, 0).numpy()  # (T, H, W, 3)

        # Postprocess and paste back
        result_frames = []
        for i in range(min(num_frames, self.chunk_size)):
            output_np = (output_video[i] * 255).astype(np.uint8)
            output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
            result = self._postprocess_frame(output_bgr, frames[i], crop_infos[i])
            result_frames.append(result)

        return result_frames

    def process_batch(self, frames: torch.Tensor, face_bboxes: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Apply de-identification to a batch of frames.

        For DeID-rPPG, this treats the batch as a video sequence.

        Args:
            frames: Batch of frames (B, C, H, W) or video (B, C, T, H, W)
            face_bboxes: Optional face bounding boxes (B, 4)
            **kwargs: Additional parameters

        Returns:
            De-identified frames tensor
        """
        if frames.dim() == 5:
            # Already a video tensor (B, C, T, H, W)
            # Normalize to [-1, 1] if not already
            if frames.max() > 1.0:
                frames = frames / 255.0
            frames = frames * 2.0 - 1.0
            frames = frames.to(self.device)

            with torch.no_grad():
                output = self.model(frames)

            # Convert back to [0, 1] or [0, 255]
            output = (output + 1.0) / 2.0
            return output.clamp(0, 1)

        else:
            # Batch of images (B, C, H, W) - use parent's default implementation
            return super().process_batch(frames, face_bboxes, **kwargs)

    def get_name(self) -> str:
        return "DeID-rPPG"

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary."""
        config = super().get_config()
        config.update({
            'method_name': 'deid_rppg',
            'chunk_size': self.chunk_size,
            'face_size': self.face_size,
            'model_path': self.model_path,
        })
        return config
