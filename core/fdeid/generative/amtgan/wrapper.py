"""
AMT-GAN: Adversarial Makeup Transfer Wrapper for the toolbox.

This wrapper provides the BaseDeIdentifier interface for AMT-GAN,
which performs face de-identification through adversarial makeup transfer.
"""

import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path

try:
    import dlib
except ImportError:
    dlib = None

# Add current directory to path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from .backbone.config import get_config
from .backbone.inference import Inference
from .backbone.postprocess import PostProcess
from ...base import BaseDeIdentifier


# Default paths
DEFAULT_GENERATOR_PATH = './weight/amtgan_data_weight/checkpoints/G.pth'
DEFAULT_REFERENCE_DIR = './weight/amtgan_data_weight/assets/datasets/reference'


class AMTGANDeIdentifier(BaseDeIdentifier):
    """
    AMT-GAN: Adversarial Makeup Transfer for Face De-identification.

    This method transfers makeup from a reference image to the source face,
    creating adversarial perturbations that protect identity while maintaining
    a natural appearance through makeup.

    Reference:
        AMT-GAN: https://github.com/CGCL-codes/AMT-GAN
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize AMT-GAN de-identifier.

        Args:
            config: Dictionary containing:
                - weights_path: Path to generator weights (G.pth)
                - reference_path: Path to reference makeup image (optional)
                - reference_dir: Directory containing reference images (optional)
                - device: Device to use (cuda/cpu)
                - dlib_path: Path to dlib landmark predictor (for masking)
        """
        super().__init__(config)

        # Use 'or' to handle both missing keys AND explicit None values
        self.weights_path = config.get('weights_path') or DEFAULT_GENERATOR_PATH
        self.reference_path = config.get('reference_path')
        self.reference_dir = config.get('reference_dir') or DEFAULT_REFERENCE_DIR
        self.dlib_path = config.get('dlib_path')

        # Get AMT-GAN config
        self.amtgan_config = get_config()
        self.amtgan_config.DEVICE.device = str(self.device)

        # Initialize inference module
        if os.path.exists(self.weights_path):
            try:
                self.inference = Inference(
                    self.amtgan_config,
                    str(self.device),
                    self.weights_path
                )
                self.postprocess = PostProcess(self.amtgan_config)
                print(f"AMT-GAN loaded from {self.weights_path}")
            except Exception as e:
                print(f"Error loading AMT-GAN: {e}")
                self.inference = None
                self.postprocess = None
        else:
            print(f"Warning: AMT-GAN weights not found at {self.weights_path}. Method will fallback.")
            self.inference = None
            self.postprocess = None

        # Load reference image(s)
        self.reference_images = []
        self._load_reference_images()

        # Initialize Dlib for masking
        self.predictor = None
        if self.dlib_path and os.path.exists(self.dlib_path) and dlib is not None:
             try:
                 self.predictor = dlib.shape_predictor(self.dlib_path)
             except Exception as e:
                 print(f"Warning: Failed to load dlib predictor: {e}")

    def _load_reference_images(self):
        """Load reference images for makeup transfer."""
        if self.reference_path and os.path.exists(self.reference_path):
            # Single reference image
            try:
                ref_img = Image.open(self.reference_path).convert('RGB')
                self.reference_images.append(ref_img)
                print(f"Loaded reference image: {self.reference_path}")
            except Exception as e:
                print(f"Error loading reference image: {e}")

        if self.reference_dir and os.path.exists(self.reference_dir):
            # Load all reference images from directory
            for fname in os.listdir(self.reference_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    fpath = os.path.join(self.reference_dir, fname)
                    try:
                        ref_img = Image.open(fpath).convert('RGB')
                        self.reference_images.append(ref_img)
                    except:
                        pass

            if self.reference_images:
                print(f"Loaded {len(self.reference_images)} reference images from {self.reference_dir}")

        if not self.reference_images:
            print("Warning: No reference images loaded. Method may use default behavior.")

    def _get_reference_image(self, index: Optional[int] = None) -> Optional[Image.Image]:
        """Get a reference image for makeup transfer."""
        if not self.reference_images:
            return None

        if index is not None and 0 <= index < len(self.reference_images):
            return self.reference_images[index]

        # Random selection
        idx = np.random.randint(0, len(self.reference_images))
        return self.reference_images[idx]

    def process_frame(self,
                     frame: np.ndarray,
                     face_bbox: Optional[Union[List, Tuple, np.ndarray]] = None,
                     **kwargs) -> np.ndarray:
        """
        Apply AMT-GAN de-identification to a frame.

        Args:
            frame: Input image as numpy array (H, W, C) in BGR format
            face_bbox: Optional face bounding box (x1, y1, x2, y2)
            **kwargs: Additional parameters:
                - reference_index: Index of reference image to use
        """
        if self.inference is None:
            # Fallback: return original or apply simple blur
            if kwargs.get('verbose', False):
                 print("Warning: AMT-GAN not initialized, returning original frame")
            return frame

        # Convert BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        source_pil = Image.fromarray(frame_rgb)

        # Get reference image
        ref_index = kwargs.get('reference_index', None)
        reference = self._get_reference_image(ref_index)

        if reference is None:
            if kwargs.get('verbose', False):
                print("Warning: No reference image available, returning original frame")
            return frame

        try:
            # Perform makeup transfer
            result, crop_face = self.inference.transfer(source_pil, reference, with_face=True)

            if result is None or crop_face is None:
                # Face not detected or transfer failed
                return frame

            # Crop source face for postprocessing
            source_crop = source_pil.crop((
                crop_face.left(),
                crop_face.top(),
                crop_face.right(),
                crop_face.bottom()
            ))

            # Postprocess
            result = self.postprocess(source_crop, result)

            # Convert result back to numpy and BGR
            if isinstance(result, Image.Image):
                result_np = np.array(result)
                result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

                # Paste result back into original frame with MASKING
                output_frame = frame.copy()

                # Get face region coordinates
                x1, y1 = max(0, crop_face.left()), max(0, crop_face.top())
                x2, y2 = min(frame.shape[1], crop_face.right()), min(frame.shape[0], crop_face.bottom())
                
                face_h = y2 - y1
                face_w = x2 - x1

                if face_h > 0 and face_w > 0:
                    # Resize generated face to match target region
                    result_resized = cv2.resize(result_bgr, (face_w, face_h))
                    
                    # Generate Mask
                    mask = None
                    
                    # Method 1: Use Dlib landmarks if available
                    if self.predictor is not None and dlib is not None:
                        try:
                            # We need to detect landmarks on the face region
                            # Construct dlib rectangle for the cropped face
                            d_rect = dlib.rectangle(x1, y1, x2, y2)
                            shape = self.predictor(frame_rgb, d_rect)
                            
                            coords = np.zeros((68, 2), dtype=int)
                            for i in range(68):
                                coords[i] = (shape.part(i).x, shape.part(i).y)
                                
                            # Create mask from Convex Hull of landmarks
                            # Adjust coords relative to the crop (x1, y1)
                            coords_crop = coords.copy()
                            coords_crop[:, 0] -= x1
                            coords_crop[:, 1] -= y1
                            
                            mask = np.zeros((face_h, face_w), dtype=np.uint8)
                            hull = cv2.convexHull(coords_crop)
                            cv2.fillConvexPoly(mask, hull, 255)
                            
                            # Soften mask
                            mask = cv2.GaussianBlur(mask, (15, 15), 10)
                            
                        except Exception as e:
                            mask = None
                            
                    # Method 2: Fallback to Ellipse Mask
                    if mask is None:
                        mask = np.zeros((face_h, face_w), dtype=np.uint8)
                        center = (face_w // 2, face_h // 2)
                        axes = (int(face_w * 0.45), int(face_h * 0.55)) # slightly smaller than box
                        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
                        # Soften
                        mask = cv2.GaussianBlur(mask, (21, 21), 11)

                    # Normalize mask to 0..1
                    mask_norm = mask.astype(float) / 255.0
                    mask_norm = np.expand_dims(mask_norm, axis=-1)

                    # Blend
                    roi = output_frame[y1:y2, x1:x2].astype(float)
                    blended = result_resized.astype(float) * mask_norm + roi * (1.0 - mask_norm)
                    
                    output_frame[y1:y2, x1:x2] = blended.astype(np.uint8)

                return output_frame
            else:
                return frame

        except Exception as e:
            print(f"Error in AMT-GAN transfer: {e}")
            import traceback
            traceback.print_exc()
            return frame

    def get_name(self) -> str:
        """Return the name of the de-identification method."""
        return "AMT-GAN"
