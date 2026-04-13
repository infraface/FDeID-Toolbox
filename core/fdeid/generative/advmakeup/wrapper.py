"""
Adv-Makeup: Adversarial Makeup Transfer Wrapper for the toolbox.

This wrapper provides the BaseDeIdentifier interface for Adv-Makeup,
which generates adversarial makeup on the eye region for de-identification.
"""

import os
import sys
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
from typing import Dict, Any, Optional, List, Union, Tuple
from torchvision.transforms import ToTensor, Normalize, Compose

# Add current directory to path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    import dlib
except ImportError:
    dlib = None

from .config import Configuration
from .networks import Encoder, Decoder
from ...base import BaseDeIdentifier


# Default paths
DEFAULT_DLIB_PATH = './weight/ciagan_pre_trained/shape_predictor_68_face_landmarks.dat'
DEFAULT_WEIGHTS_DIR = './weight/advmakeup_pre_trained'
DEFAULT_TARGET_NAME = '00288'


class AdvMakeupDeIdentifier(BaseDeIdentifier):
    """
    Adv-Makeup: Adversarial Makeup Transfer for Face De-identification.

    This method applies adversarial makeup to the eye region to protect
    facial identity while maintaining a natural appearance.

    Reference:
        Adv-Makeup: https://github.com/TencentYoutuResearch/Adv-Makeup
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Adv-Makeup de-identifier.

        Args:
            config: Dictionary containing:
                - weights_path: Path to model weights directory
                - target_name: Target identity name for makeup style (default: '00288')
                - dlib_path: Path to dlib landmark predictor
                - device: Device to use (cuda/cpu)
        """
        super().__init__(config)

        if dlib is None:
            raise ImportError(
                "dlib is required for Adv-Makeup landmarks. Please install it with 'pip install dlib'."
            )

        # Use 'or' to handle both missing keys and explicit None values
        self.weights_dir = config.get('weights_path') or DEFAULT_WEIGHTS_DIR
        self.target_name = config.get('target_name') or '00288'
        self.dlib_path = config.get('dlib_path') or DEFAULT_DLIB_PATH

        # Load Adv-Makeup Config
        self.cfg = Configuration()

        # Set device in config based on wrapper device setting
        if self.device == 'cpu' or self.device == torch.device('cpu'):
            self.cfg.gpu = -1  # -1 means CPU in MakeupAttack
        else:
            # Extract GPU index if device is like 'cuda:0'
            if isinstance(self.device, str) and 'cuda' in self.device:
                if ':' in self.device:
                    self.cfg.gpu = int(self.device.split(':')[1])
                else:
                    self.cfg.gpu = 0
            elif isinstance(self.device, torch.device):
                self.cfg.gpu = self.device.index if self.device.index is not None else 0
            else:
                self.cfg.gpu = 0

        # Initialize Model
        self.model = None
        self._load_model()

        # Preprocessing transforms
        self.transforms = Compose([
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Initialize Dlib
        if not os.path.exists(self.dlib_path):
            print(f"Warning: Dlib model not found at {self.dlib_path}")
            self.detector = None
            self.predictor = None
        else:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(self.dlib_path)

    def _load_model(self):
        """Load Adv-Makeup model weights (encoder and decoder only for inference)."""
        try:
            # Construct paths for encoder and decoder
            target_id = self.target_name.split('.')[0] if '.' in self.target_name else self.target_name
            weights_subdir = os.path.join(self.weights_dir, target_id)

            model_id = self.cfg.epoch_steps - 1
            enc_path = os.path.join(weights_subdir, f'{model_id:05d}_enc.pth')
            dec_path = os.path.join(weights_subdir, f'{model_id:05d}_dec.pth')

            if os.path.exists(enc_path) and os.path.exists(dec_path):
                print(f"Loading Adv-Makeup weights from {weights_subdir}")

                # Create encoder and decoder directly (inference-only mode)
                self.enc = Encoder(self.cfg.input_dim)
                self.dec = Decoder(self.cfg.input_dim)

                # Load weights (use weights_only=True for security)
                device_str = str(self.device) if isinstance(self.device, torch.device) else self.device
                self.enc.load_state_dict(torch.load(enc_path, map_location=device_str, weights_only=True))
                self.dec.load_state_dict(torch.load(dec_path, map_location=device_str, weights_only=True))

                # Move to device and set eval mode
                self.enc.to(self.device).eval()
                self.dec.to(self.device).eval()

                # Set model flag to indicate successful loading
                self.model = True
                print(f"Adv-Makeup encoder and decoder loaded successfully")
            else:
                print(f"Warning: Adv-Makeup weights not found at {weights_subdir}")
                print(f"  Expected: {enc_path}")
                print(f"  Expected: {dec_path}")
                self.model = None

        except Exception as e:
            print(f"Error loading Adv-Makeup model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None

    def get_landmarks(self, img_rgb: np.ndarray, face_bbox: Optional[Union[List, Tuple, np.ndarray]] = None) -> Optional[np.ndarray]:
        """
        Get 68 facial landmarks using dlib.

        Args:
            img_rgb: RGB image (H, W, 3) uint8
            face_bbox: Optional face bounding box (x1, y1, x2, y2)

        Returns:
            Landmarks array (68, 2) or None if no face detected
        """
        if self.predictor is None:
            return None

        if face_bbox is not None:
            # Use provided bounding box
            try:
                x1, y1, x2, y2 = face_bbox
                d = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))
            except Exception as e:
                # If bbox format is wrong, fall back to detector
                if self.detector is None:
                    return None
                dets = self.detector(img_rgb, 1)
                if len(dets) == 0:
                    return None
                d = dets[0]
        else:
            # Use detector
            if self.detector is None:
                return None
            dets = self.detector(img_rgb, 1)
            if len(dets) == 0:
                return None
            d = dets[0]

        shape = self.predictor(img_rgb, d)
        coords = np.zeros((68, 2), dtype=int)
        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def process_frame(self,
                     frame: np.ndarray,
                     face_bbox: Optional[Union[List, Tuple, np.ndarray]] = None,
                     **kwargs) -> np.ndarray:
        """
        Apply Adv-Makeup de-identification to a frame.

        Args:
            frame: Input image (H, W, C) in BGR format
            face_bbox: Optional face bounding box (x1, y1, x2, y2)
            **kwargs: Additional parameters

        Returns:
            De-identified frame in BGR format
        """
        if self.model is None:
            print("Warning: Adv-Makeup model not loaded, returning original frame")
            return frame

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get facial landmarks using dlib
        lmks = self.get_landmarks(img_rgb, face_bbox)

        if lmks is None:
            # Try once more without bbox if failed (maybe bbox was bad?)
            if face_bbox is not None:
                lmks = self.get_landmarks(img_rgb, None)
            
            if lmks is None:
                return frame

        # Define Eye Area using Dlib 68 points
        # Left Eye: 36-41, Right Eye: 42-47
        # Left Brow: 17-21, Right Brow: 22-26

        # Calculate bounding box of eye features
        eye_feats = np.concatenate([
            lmks[17:22],  # Left Brow
            lmks[22:27],  # Right Brow
            lmks[36:42],  # Left Eye
            lmks[42:48],  # Right Eye
            lmks[27:30]   # Nose bridge
        ], axis=0)

        x_min, y_min = eye_feats.min(axis=0)
        x_max, y_max = eye_feats.max(axis=0)

        # Calculate crop dimensions
        # Adv-Makeup expects (420, 160) input
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2

        jaw_width = lmks[16, 0] - lmks[0, 0]
        crop_w = int(jaw_width * 1.1)
        crop_h = int(crop_w / 420 * 160)

        # Center crop around eyes
        x1 = int(cx - crop_w / 2)
        x2 = int(cx + crop_w / 2)
        y1 = int(cy - crop_h / 2)
        y2 = int(cy + crop_h / 2)

        # Pad coordinates
        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, x2 - img_rgb.shape[1])
        pad_b = max(0, y2 - img_rgb.shape[0])

        img_padded = cv2.copyMakeBorder(
            img_rgb, pad_t, pad_b, pad_l, pad_r,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        # Adjust coords for padding
        x1 += pad_l
        x2 += pad_l
        y1 += pad_t
        y2 += pad_t

        crop_img = img_padded[y1:y2, x1:x2]
        crop_img_pil = Image.fromarray(crop_img).resize((420, 160), Image.BILINEAR)

        # Prepare Tensor
        inp = self.transforms(crop_img_pil).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            fake_content = self.enc(inp, bn_training=False)
            fake_after = self.dec(*fake_content, bn_training=False)

        # Get actual crop dimensions
        crop_h_real, crop_w_real = crop_img.shape[:2]

        # Resize output back to actual crop size
        fake_after_img = F.interpolate(fake_after, size=(crop_h_real, crop_w_real), mode='bilinear')
        fake_after_np = fake_after_img.squeeze(0).cpu().permute(1, 2, 0).numpy()

        # Denormalize: (x * 0.5 + 0.5) * 255
        fake_after_np = ((fake_after_np * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

        # Generate Blending Mask
        # Convert landmarks from original image coords to crop coords
        # In padded image, landmarks are at: lmks + (pad_l, pad_t)
        # Relative to crop start (x1, y1): (lmks + pad) - (x1, y1)
        lmks_crop = lmks.copy()
        lmks_crop[:, 0] += pad_l - x1  # = lmks_x + pad_l - x1
        lmks_crop[:, 1] += pad_t - y1  # = lmks_y + pad_t - y1

        mask_h, mask_w = crop_h_real, crop_w_real
        mask_img = Image.new('L', (mask_w, mask_h), 0)
        draw = ImageDraw.Draw(mask_img)

        # Left Region (Eye + Brow)
        l_pts = np.concatenate([lmks_crop[17:22], lmks_crop[36:42]])
        l_hull = cv2.convexHull(l_pts.astype(np.int32))
        draw.polygon([tuple(p[0]) for p in l_hull], fill=255)

        # Right Region
        r_pts = np.concatenate([lmks_crop[22:27], lmks_crop[42:48]])
        r_hull = cv2.convexHull(r_pts.astype(np.int32))
        draw.polygon([tuple(p[0]) for p in r_hull], fill=255)

        # Exclude Eye-Balls
        l_eye_poly = lmks_crop[36:42]
        draw.polygon([tuple(p) for p in l_eye_poly], fill=0)
        r_eye_poly = lmks_crop[42:48]
        draw.polygon([tuple(p) for p in r_eye_poly], fill=0)

        # Smooth mask
        mask_np = np.array(mask_img)
        mask_np = cv2.GaussianBlur(mask_np, (15, 15), 0)
        mask_np = mask_np / 255.0
        mask_np = np.expand_dims(mask_np, axis=-1)

        # Blend
        combined_crop = fake_after_np * mask_np + crop_img * (1 - mask_np)

        # Put back
        img_padded[y1:y2, x1:x2] = combined_crop.astype(np.uint8)

        # Unpad
        result = img_padded[pad_t:pad_t + img_rgb.shape[0], pad_l:pad_l + img_rgb.shape[1]]

        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    def get_name(self) -> str:
        """Return the name of the de-identification method."""
        return "Adv-Makeup"
