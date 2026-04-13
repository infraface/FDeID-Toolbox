"""
G2Face: High-Fidelity Reversible Face De-identification.

Wrapper for the G2Face implementation integrated into the toolbox.
G2Face uses StyleGAN2-based generator with identity-aware feature fusion
and 3D face geometric priors for high-quality face anonymization.

Key features:
- High-fidelity anonymization with new identity synthesis
- Geometry preservation via BFM 3D face model
- Reversible: original identity can be recovered from anonymized image
"""

import os
import sys
import cv2
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from torchvision import transforms
from argparse import Namespace

# Add local model path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from ...base import BaseDeIdentifier

# Default paths
DEFAULT_G2FACE_WEIGHTS = './weight/G2Face_pre_trained/weights/G2Face.pth'
DEFAULT_ARCFACE_WEIGHTS = './weight/G2Face_pre_trained/weights/ms1mv3_arcface_r50.pth'
DEFAULT_RECON_WEIGHTS = './weight/G2Face_pre_trained/weights/epoch_20.pth'
DEFAULT_BFM_FOLDER = './weight/G2Face_pre_trained/BFM'


def get_default_opt():
    """Get default options for G2Face models."""
    opt = Namespace()
    # Model structure
    opt.image_size = 256
    opt.style_dim = 512
    opt.latent_dim = 659  # 512 + 144 + 3
    opt.n_mlp = 4
    opt.device = 'cuda'

    # 3D face reconstruction
    opt.net_recon = 'resnet50'
    opt.use_last_fc = False
    opt.init_path = DEFAULT_RECON_WEIGHTS
    opt.bfm_folder = DEFAULT_BFM_FOLDER
    opt.bfm_model = 'BFM_model_front.mat'
    opt.checkpoints_dir = os.path.dirname(DEFAULT_RECON_WEIGHTS)
    opt.name = ''
    opt.isTrain = False
    opt.phase = 'test'

    # Renderer parameters
    opt.focal = 1015.0
    opt.center = 112.0
    opt.camera_d = 10.0
    opt.z_near = 5.0
    opt.z_far = 15.0
    opt.use_opengl = True

    return opt


class G2FaceDeIdentifier(BaseDeIdentifier):
    """
    G2Face: High-Fidelity Reversible Face De-identification.

    Uses StyleGAN2-based generator with:
    - Identity-aware feature fusion (IFF)
    - 3D face geometric priors (BFM model)
    - Information hiding for identity recovery

    Input: 256x256 aligned face images
    Output: 256x256 anonymized face images
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Configuration
        self.model_path = config.get('weights_path') or config.get('model_path') or DEFAULT_G2FACE_WEIGHTS
        self.arcface_path = config.get('arcface_path') or DEFAULT_ARCFACE_WEIGHTS
        self.recon_path = config.get('recon_path') or DEFAULT_RECON_WEIGHTS
        self.bfm_folder = config.get('bfm_folder') or DEFAULT_BFM_FOLDER
        self.image_size = config.get('image_size', 256)

        # Get options
        self.opt = get_default_opt()
        self.opt.device = str(self.device)
        self.opt.init_path = self.recon_path
        self.opt.bfm_folder = self.bfm_folder
        self.opt.checkpoints_dir = os.path.dirname(self.recon_path)

        # Verify weights exist
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"G2Face weights not found at '{self.model_path}'.\n"
                f"Please ensure the pretrained weights are available."
            )

        # Import models
        from .model import Generator, HidingExtractor, MLP, Map2ID, FaceReconModel, iresnet50
        from .utils.binary_converter import float2bit, bit2float

        self.float2bit = float2bit
        self.bit2float = bit2float

        # Load checkpoint
        print(f"Loading G2Face checkpoint from {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)

        # Initialize Generator
        self.generator = Generator(size=self.image_size, style_dim=self.opt.style_dim)
        self.generator.load_state_dict(checkpoint['anonymous_net'])
        self.generator.to(self.device).eval()

        # Initialize StyleMLP
        self.style_mlp = MLP(
            latent_dim=self.opt.latent_dim,
            style_dim=self.opt.style_dim,
            n_mlp=self.opt.n_mlp
        )
        self.style_mlp.load_state_dict(checkpoint['style_mlp'])
        self.style_mlp.to(self.device).eval()

        # Initialize Map2ID
        self.map_2_id = Map2ID()
        self.map_2_id.load_state_dict(checkpoint['map_2_id'])
        self.map_2_id.to(self.device).eval()

        # Initialize HidingExtractor (for reversibility)
        self.hiding_extractor = HidingExtractor()
        self.hiding_extractor.load_state_dict(checkpoint['hiding_extractor'])
        self.hiding_extractor.to(self.device).eval()

        # Initialize ArcFace
        self.arc_face = iresnet50()
        if os.path.exists(self.arcface_path):
            self.arc_face.load_state_dict(torch.load(self.arcface_path, map_location='cpu', weights_only=True))
        else:
            print(f"Warning: ArcFace weights not found at {self.arcface_path}")
        self.arc_face.to(self.device).eval()

        # Initialize 3D Face Reconstruction
        # Note: use_renderer=False because nvdiffrast only works on NVIDIA CUDA
        # For inference, we only need compute_coeff() which doesn't require the renderer
        try:
            self.face_recon = FaceReconModel(self.opt, use_renderer=False)
            self.face_recon.to(self.device)
            self.face_recon.load_networks(20)
            self.face_recon.eval()
            self._use_3d_recon = True
        except Exception as e:
            print(f"Warning: Failed to initialize 3D face reconstruction: {e}")
            print("G2Face will use simplified mode without geometry preservation.")
            self._use_3d_recon = False

        # Image transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # Denormalizer
        self.denorm_mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(self.device)
        self.denorm_std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(self.device)

        print("G2Face initialized successfully")

    def _denormalize(self, x):
        """Denormalize from [-1, 1] to [0, 1]."""
        return x * self.denorm_std + self.denorm_mean

    def _preprocess(self, frame: np.ndarray, face_bbox: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        Preprocess frame for G2Face.

        Args:
            frame: Input frame (H, W, C) in BGR format
            face_bbox: Optional face bounding box (x1, y1, x2, y2)

        Returns:
            tensor: Preprocessed tensor (1, 3, 256, 256) in [-1, 1]
            crop_info: (x1, y1, x2, y2, orig_h, orig_w) for restoration
        """
        h, w = frame.shape[:2]

        if face_bbox is not None:
            x1, y1, x2, y2 = map(int, face_bbox)
        else:
            # Use full image if no bbox
            x1, y1, x2, y2 = 0, 0, w, h

        # Ensure valid coordinates
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # Crop and resize
        face_crop = frame[y1:y2, x1:x2].copy()
        orig_h, orig_w = face_crop.shape[:2]

        if face_crop.size == 0:
            face_crop = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            orig_h, orig_w = self.image_size, self.image_size

        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        # Apply transforms
        tensor = self.transform(face_rgb).unsqueeze(0).to(self.device)

        return tensor, (x1, y1, x2, y2, orig_h, orig_w)

    def _postprocess(self, output: torch.Tensor, original_frame: np.ndarray, crop_info: Tuple) -> np.ndarray:
        """
        Postprocess output and paste back to original frame.

        Args:
            output: Output tensor (1, 3, 256, 256) in [-1, 1]
            original_frame: Original frame to paste into
            crop_info: (x1, y1, x2, y2, orig_h, orig_w)

        Returns:
            Frame with anonymized face
        """
        x1, y1, x2, y2, orig_h, orig_w = crop_info
        result = original_frame.copy()

        # Convert tensor to numpy
        output = output.clamp(-1, 1)
        output_np = ((output[0] + 1) / 2 * 255).cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)

        # Resize back to original crop size
        if orig_h > 0 and orig_w > 0:
            output_resized = cv2.resize(output_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            result[y1:y2, x1:x2] = output_resized

        return result

    def process_frame(self, frame: np.ndarray, face_bbox: Optional[Union[List, Tuple, np.ndarray]] = None, **kwargs) -> np.ndarray:
        """
        Apply G2Face de-identification to a single frame.

        Args:
            frame: Input image (H, W, C) in BGR format, range [0, 255]
            face_bbox: Optional face bounding box (x1, y1, x2, y2)
            **kwargs: Additional parameters
                - random_seed: Seed for random identity generation

        Returns:
            De-identified frame
        """
        # Preprocess
        input_tensor, crop_info = self._preprocess(frame, face_bbox)

        with torch.no_grad():
            # Extract identity features
            latent_face = self.arc_face(input_tensor)  # [1, 512]

            # Extract 3D geometry
            if self._use_3d_recon:
                denorm_input = self._denormalize(input_tensor)
                latent_3d, _ = self.face_recon.compute_coeff(denorm_input)  # [1, 257]
                # Extract geometry coefficients: id (80) + exp (64) = 144, pose (3)
                geometry = torch.cat([latent_3d[:, :144], latent_3d[:, 224:227]], dim=1)  # [1, 147]
            else:
                # Use zeros if 3D reconstruction not available
                geometry = torch.zeros(1, 147, device=self.device)

            # Generate random identity
            random_seed = kwargs.get('random_seed', None)
            if random_seed is not None:
                torch.manual_seed(random_seed)
            rand_z = torch.randn(1, 512, device=self.device)
            rand_id = self.map_2_id(rand_z)  # [1, 512]

            # Convert original identity to binary for hiding
            latent_hiding = self.float2bit(latent_face)  # [1, 512, 32]

            # Generate style codes
            latent_control = self.style_mlp(rand_id, geometry)  # [1, 14, 512]

            # Generate anonymized face
            anonymous_output = self.generator(input_tensor, latent_control, latent_hiding)

        # Postprocess
        result = self._postprocess(anonymous_output, frame, crop_info)

        return result

    def recover_identity(self, anonymous_frame: np.ndarray, face_bbox: Optional[Union[List, Tuple, np.ndarray]] = None) -> np.ndarray:
        """
        Recover original identity from anonymized frame.

        G2Face embeds the original identity information in the anonymized image,
        allowing recovery of the original appearance.

        Args:
            anonymous_frame: Anonymized image (H, W, C) in BGR format
            face_bbox: Optional face bounding box

        Returns:
            Recovered frame with original identity
        """
        # Preprocess
        input_tensor, crop_info = self._preprocess(anonymous_frame, face_bbox)

        with torch.no_grad():
            # Extract hidden identity information
            extract_info_bin = self.hiding_extractor(input_tensor)  # [1, 512, 32]

            # Convert binary to float
            extract_id = torch.round(torch.sigmoid(extract_info_bin))
            extract_id_float = self.bit2float(extract_id.cpu()).to(self.device)

            # Clamp extreme values
            extract_id_float = torch.clamp(extract_id_float, -50, 50)
            extract_id_float = torch.where(
                torch.abs(extract_id_float) > 50,
                torch.zeros_like(extract_id_float),
                extract_id_float
            )

            # Get geometry from anonymous image
            if self._use_3d_recon:
                denorm_input = self._denormalize(input_tensor)
                latent_3d, _ = self.face_recon.compute_coeff(denorm_input)
                geometry = torch.cat([latent_3d[:, :144], latent_3d[:, 224:227]], dim=1)
            else:
                geometry = torch.zeros(1, 147, device=self.device)

            # Generate recovery style codes
            latent_control_recover = self.style_mlp(extract_id_float, geometry)

            # Create hiding tensor (use extracted info)
            latent_hiding = self.float2bit(extract_id_float)

            # Generate recovered face
            recovered_output = self.generator(input_tensor, latent_control_recover, latent_hiding)

        # Postprocess
        result = self._postprocess(recovered_output, anonymous_frame, crop_info)

        return result

    def get_name(self) -> str:
        return "G2Face"

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary."""
        config = super().get_config()
        config.update({
            'method_name': 'g2face',
            'image_size': self.image_size,
            'model_path': self.model_path,
            'use_3d_recon': self._use_3d_recon,
        })
        return config
