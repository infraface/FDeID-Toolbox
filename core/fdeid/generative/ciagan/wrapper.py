"""
CIAGAN: Conditional Identity Anonymization Generative Adversarial Networks.
Wrapper for the official implementation integrated into the toolbox.
"""

import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, List, Union, Tuple
from torchvision import transforms

# Ensure local imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    import dlib
except ImportError:
    dlib = None

from .arch import arch_unet_flex as arch_gen
from ...base import BaseDeIdentifier
from ....identity.retinaface import FaceDetector


# Default paths (overridable via config)
DEFAULT_RETINAFACE_MODEL = './weight/retinaface_pre_trained/Resnet50_Final.pth'
DEFAULT_DLIB_LANDMARK_MODEL = './weight/ciagan_pre_trained/shape_predictor_68_face_landmarks.dat'
DEFAULT_CIAGAN_WEIGHTS = './weight/ciagan_pre_trained/modelG_ciagan.pth'


class CIAGANDeIdentifier(BaseDeIdentifier):
    """
    CIAGAN: Conditional Identity Anonymization Generative Adversarial Networks.
    Wrapper for the official implementation using RetinaFace for detection and Dlib for landmarks.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        if dlib is None:
            raise ImportError(
                "dlib is required for CIAGAN landmarks. Please install it with 'pip install dlib'."
            )

        # Use 'or' to handle both missing keys AND explicit None values
        self.model_path = config.get('weights_path') or DEFAULT_CIAGAN_WEIGHTS
        self.dlib_path = config.get('dlib_path') or DEFAULT_DLIB_LANDMARK_MODEL
        self.retinaface_path = config.get('retinaface_path') or DEFAULT_RETINAFACE_MODEL

        # Initialize RetinaFace Detector
        if not os.path.exists(self.retinaface_path):
            print(f"Warning: RetinaFace weights not found at {self.retinaface_path}. Detection might fail.")

        try:
            self.detector = FaceDetector(
                model_path=self.retinaface_path,
                network='resnet50',
                device=self.device
            )
        except Exception as e:
            print(f"Failed to initialize RetinaFace: {e}. Falling back to Dlib detector.")
            self.detector = None
            self.dlib_detector = dlib.get_frontal_face_detector()

        # Initialize Dlib Predictor
        if not os.path.exists(self.dlib_path):
            raise FileNotFoundError(f"Dlib shape predictor not found at {self.dlib_path}.")

        try:
            self.predictor = dlib.shape_predictor(self.dlib_path)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize dlib predictor: {e}")

        # Initialize CIAGAN Model
        self.generator = arch_gen.Generator()
        if self.model_path and os.path.exists(self.model_path):
            try:
                state_dict = torch.load(self.model_path, map_location='cpu', weights_only=False)
                self.generator.load_state_dict(state_dict)
                print(f"CIAGAN weights loaded from {self.model_path}")
            except Exception as e:
                print(f"Error loading CIAGAN weights: {e}")
        else:
            print(f"Warning: CIAGAN weights not found at '{self.model_path}'. Using random weights.")

        self.generator.to(self.device).eval()

        self.transform = transforms.Compose([transforms.ToTensor()])
        # CRITICAL: The CIAGAN model was trained on 128x128 images
        # The original process_data.py creates 178x218 preprocessed images,
        # but util_data.py resizes them to 128x128 before feeding to the model
        self.model_size = 128  # Model input size (square)
        self.res_w = 178  # Intermediate processing size (kept for landmark calculations)
        self.res_h = 218
        self.line_px = 1

    def _draw_landmarks(self, img_shape, landmarks, t_x, t_y, ratio_w, ratio_h):
        img_lndm = np.ones((self.res_h, self.res_w, 3), np.uint8) * 255

        def draw_line(offset, pt_st, pt_end):
            pt1 = (int((landmarks.part(offset + pt_st).x - t_x) * ratio_w),
                   int((landmarks.part(offset + pt_st).y - t_y) * ratio_h))
            pt2 = (int((landmarks.part(offset + pt_end).x - t_x) * ratio_w),
                   int((landmarks.part(offset + pt_end).y - t_y) * ratio_h))
            cv2.line(img_lndm, pt1, pt2, (0, 0, 255), self.line_px)

        for i in range(16): draw_line(0, i, i+1)
        for i in range(3): draw_line(27, i, i+1)
        for i in range(4): draw_line(60, i, i+1)
        for i in range(3): draw_line(64, i, i+1)
        draw_line(0, 67, 60)

        return img_lndm

    def _create_mask(self, img_shape, landmarks, t_x, t_y, ratio_w, ratio_h):
        img_msk = np.ones((self.res_h, self.res_w, 3), np.uint8) * 255
        contours = np.zeros((0, 2))

        p0 = [(landmarks.part(0).x - t_x) * ratio_w, (landmarks.part(19).y - t_y) * ratio_h]
        contours = np.concatenate((contours, [p0]), axis=0)

        for p in range(17):
            pt = [(landmarks.part(p).x - t_x) * ratio_w, (landmarks.part(p).y - t_y) * ratio_h]
            contours = np.concatenate((contours, [pt]), axis=0)

        p_last = [(landmarks.part(16).x - t_x) * ratio_w, (landmarks.part(24).y - t_y) * ratio_h]
        contours = np.concatenate((contours, [p_last]), axis=0)

        contours = contours.astype(int)
        cv2.fillPoly(img_msk, pts=[contours], color=(0, 0, 0))

        return img_msk

    def process_frame(self, frame: np.ndarray, face_bbox: Optional[Union[List, Tuple, np.ndarray]] = None, **kwargs) -> np.ndarray:
        result_frame = frame.copy()

        # Detection
        dlib_rects = []
        if face_bbox is not None:
            x1, y1, x2, y2 = map(int, face_bbox)
            dlib_rects = [dlib.rectangle(max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2))]
        else:
            if self.detector:
                detections = self.detector.detect(frame)
                for det in detections:
                    if det.confidence < 0.5:
                        continue
                    b = det.bbox.astype(int)
                    dlib_rects.append(dlib.rectangle(
                        max(0, b[0]), max(0, b[1]),
                        min(frame.shape[1], b[2]), min(frame.shape[0], b[3])
                    ))
            elif hasattr(self, 'dlib_detector'):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                dlib_rects = self.dlib_detector(gray, 1)

        if not dlib_rects:
            return result_frame

        # Processing loop
        for d in dlib_rects:
            try:
                landmarks = self.predictor(frame, d)

                # Alignment logic
                c_x = int((landmarks.part(42).x + landmarks.part(39).x) / 2)
                c_y = int((landmarks.part(42).y + landmarks.part(39).y) / 2)
                eye_w = landmarks.part(42).x - landmarks.part(39).x
                w_r = int(eye_w * 4)
                h_r = int(eye_w * 5)

                if h_r == 0:
                    continue
                w_r = int(h_r / self.res_h * self.res_w)

                w, h = int(w_r * 2), int(h_r * 2)
                if w <= 0 or h <= 0:
                    continue

                pd = int(w)

                # Pad image
                img_p = np.full((frame.shape[0]+pd*2, frame.shape[1]+pd*2, 3), 255, dtype=np.uint8)
                img_p[pd:pd+frame.shape[0], pd:pd+frame.shape[1]] = frame

                c_x_p = c_x + pd
                c_y_p = c_y + pd

                # Crop
                visual = img_p[c_y_p - h_r:c_y_p + h_r, c_x_p - w_r:c_x_p + w_r]

                if visual.size == 0 or visual.shape[0] != h or visual.shape[1] != w:
                    continue

                # Resize to intermediate processing size (178x218)
                ratio_w, ratio_h = self.res_w/w, self.res_h/h
                visual_resized = cv2.resize(visual, (self.res_w, self.res_h), interpolation=cv2.INTER_CUBIC)

                t_x = c_x_p - w_r
                t_y = c_y_p - h_r

                offset_x = t_x - pd
                offset_y = t_y - pd

                # Create landmarks and mask at intermediate size (178x218)
                img_lndm = self._draw_landmarks(None, landmarks, offset_x, offset_y, ratio_w, ratio_h)
                img_msk = self._create_mask(None, landmarks, offset_x, offset_y, ratio_w, ratio_h)

                # CRITICAL: Resize to model input size (128x128) - matching training pipeline
                model_size = self.model_size
                visual_model = cv2.resize(visual_resized, (model_size, model_size), interpolation=cv2.INTER_CUBIC)
                img_lndm_model = cv2.resize(img_lndm, (model_size, model_size), interpolation=cv2.INTER_NEAREST)
                img_msk_model = cv2.resize(img_msk, (model_size, model_size), interpolation=cv2.INTER_NEAREST)

                # Prepare Tensors at model size (128x128)
                visual_rgb = cv2.cvtColor(visual_model, cv2.COLOR_BGR2RGB)
                im_faces = self.transform(visual_rgb).float().unsqueeze(0).to(self.device)

                im_lndm_tensor = self.transform(cv2.cvtColor(img_lndm_model, cv2.COLOR_BGR2RGB)).float().unsqueeze(0).to(self.device)

                im_msk_tensor = self.transform(cv2.cvtColor(img_msk_model, cv2.COLOR_BGR2RGB))
                im_msk_tensor = ((1 - im_msk_tensor) > 0.2).float().to(self.device).unsqueeze(0)

                # Label
                target_id = kwargs.get('target_id', np.random.randint(0, 1200))
                labels_one_hot = torch.zeros((1, 1200), device=self.device)
                labels_one_hot[0, int(target_id) % 1200] = 1

                # Inference
                with torch.no_grad():
                    input_gen = torch.cat((im_lndm_tensor, im_faces * (1 - im_msk_tensor)), 1)
                    im_gen = self.generator(input_gen, onehot=labels_one_hot)
                    im_gen = torch.clamp(im_gen*im_msk_tensor + im_faces * (1 - im_msk_tensor), 0, 1)

                # Restore: output is 128x128, resize to original face crop size
                im_gen_np = (im_gen[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                im_gen_bgr = cv2.cvtColor(im_gen_np, cv2.COLOR_RGB2BGR)
                im_gen_restored = cv2.resize(im_gen_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

                # Paste
                fy1 = max(0, c_y - h_r)
                fy2 = min(frame.shape[0], c_y + h_r)
                fx1 = max(0, c_x - w_r)
                fx2 = min(frame.shape[1], c_x + w_r)

                gy1 = fy1 - (c_y - h_r)
                gy2 = gy1 + (fy2 - fy1)
                gx1 = fx1 - (c_x - w_r)
                gx2 = gx1 + (fx2 - fx1)

                if fy2 > fy1 and fx2 > fx1:
                    result_frame[fy1:fy2, fx1:fx2] = im_gen_restored[gy1:gy2, gx1:gx2]

            except Exception as e:
                # Silently skip faces that fail processing
                # Uncomment for debugging: print(f"CIAGAN processing error for face: {e}")
                continue

        return result_frame
