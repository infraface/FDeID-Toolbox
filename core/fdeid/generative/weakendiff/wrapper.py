"""
WeakenDiff: Diffusion-based Face De-identification Wrapper for the toolbox.

This wrapper provides the BaseDeIdentifier interface for WeakenDiff,
which uses Stable Diffusion with adversarial optimization for de-identification.
"""

import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, List, Union, Tuple
from torchvision import transforms
from tqdm import tqdm

# Add current directory to path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from ...base import BaseDeIdentifier
from .adversarial_optimization import Adversarial_Opt
from .utils import get_target_test_images, alignment
from .attention_control import AttentionControlEdit

# Default paths
DEFAULT_DIFFUSION_MODEL = './weight/weakendiff_pre_trained'
DEFAULT_ASSETS_PATH = os.path.join(current_dir, 'assets')


class WeakenDiffConfig:
    """Configuration class for WeakenDiff parameters."""

    # Presets for different speed/quality tradeoffs
    PRESETS = {
        'fast': {
            'prot_steps': 10,
            'diffusion_steps': 10,
            'start_step': 7,
            'null_optimization_steps': 5,
        },
        'balanced': {
            'prot_steps': 15,
            'diffusion_steps': 15,
            'start_step': 12,
            'null_optimization_steps': 10,
        },
        'quality': {
            'prot_steps': 30,
            'diffusion_steps': 20,
            'start_step': 17,
            'null_optimization_steps': 20,
        },
    }

    def __init__(self, **kwargs):
        # Check for preset
        preset = kwargs.get('preset', None)
        if preset and preset in self.PRESETS:
            preset_config = self.PRESETS[preset]
            for k, v in preset_config.items():
                if k not in kwargs:  # Only use preset if not explicitly set
                    kwargs[k] = v

        # Default parameters from the original implementation
        self.image_size = kwargs.get('image_size', 256)
        self.prot_steps = kwargs.get('prot_steps', 30)
        self.diffusion_steps = kwargs.get('diffusion_steps', 20)
        self.start_step = kwargs.get('start_step', 17)
        self.null_optimization_steps = kwargs.get('null_optimization_steps', 20)
        self.adv_optim_weight = kwargs.get('adv_optim_weight', 0.003)
        self.makeup_weight = kwargs.get('makeup_weight', 0)
        self.is_makeup = kwargs.get('is_makeup', False)
        self.is_obfuscation = kwargs.get('is_obfuscation', True)
        self.MTCNN_cropping = kwargs.get('MTCNN_cropping', True)
        # Using '2' as default target (usually a generic identity)
        self.target_choice = kwargs.get('target_choice', '2')
        self.source_text = kwargs.get('source_text', 'face')
        self.makeup_prompt = kwargs.get('makeup_prompt', 'red lipstick')
        self.comparison_null_text = kwargs.get('comparison_null_text', False)
        self.surrogate_model_names = kwargs.get('surrogate_model_names', ['ir152', 'facenet', 'irse50'])
        self.test_model_name = kwargs.get('test_model_name', ['mobile_face'])
        self.protected_image_dir = kwargs.get('save_dir', '/tmp')
        self.device = kwargs.get('device', 'cuda')
        self.assets_path = kwargs.get('assets_path')
        self.dataloader = None
        self.source_dir = kwargs.get('source_dir', '')

        # Performance options
        # Disable fp16 by default - mixed precision causes numerical instability
        # during adversarial optimization (gradient underflow, dtype mismatches)
        self.use_fp16 = kwargs.get('use_fp16', False)
        self.use_compile = kwargs.get('use_compile', False)  # torch.compile() - may have issues on some systems


class WeakenDiffDeIdentifier(BaseDeIdentifier):
    """
    WeakenDiff: Diffusion-based Face De-identification.

    This method uses Stable Diffusion with adversarial optimization.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.diffusion_model_id = config.get('diffusion_model_id', DEFAULT_DIFFUSION_MODEL)
        self.assets_path = config.get('assets_path', DEFAULT_ASSETS_PATH)

        # Create WeakenDiff config
        self.wdiff_config = WeakenDiffConfig(**config)
        self.wdiff_config.device = self.device
        self.wdiff_config.assets_path = self.assets_path
        
        # Override save_dir with config's output if available, else tmp
        # Adversarial_Opt uses protected_image_dir for logging null text
        
        self._diff_model = None
        self._attacker_model = None # This will hold Adversarial_Opt instance
        self._initialized = False

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((self.wdiff_config.image_size, self.wdiff_config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def _lazy_init(self):
        """Lazy initialization of heavy models."""
        if self._initialized:
            return

        print("Loading Stable Diffusion model and FR models... (this may take a while)")
        print(f"Performance settings: fp16={self.wdiff_config.use_fp16}, "
              f"prot_steps={self.wdiff_config.prot_steps}, "
              f"diffusion_steps={self.wdiff_config.diffusion_steps}, "
              f"null_opt_steps={self.wdiff_config.null_optimization_steps}")

        try:
            from diffusers import StableDiffusionPipeline, DDIMScheduler

            # Load Stable Diffusion
            model_id = self.diffusion_model_id

            # Determine dtype based on fp16 setting
            torch_dtype = torch.float16 if self.wdiff_config.use_fp16 else torch.float32

            # Check for local directory/single file
            load_from_single_file = os.path.isfile(model_id) or model_id.endswith('.safetensors') or model_id.endswith('.ckpt')

            if load_from_single_file:
                self._diff_model = StableDiffusionPipeline.from_single_file(
                    model_id, local_files_only=True, torch_dtype=torch_dtype
                ).to(self.device)
            else:
                 # Check if local dir
                if os.path.exists(model_id):
                    self._diff_model = StableDiffusionPipeline.from_pretrained(
                        model_id, local_files_only=True, torch_dtype=torch_dtype
                    ).to(self.device)
                else:
                    self._diff_model = StableDiffusionPipeline.from_pretrained(
                       model_id, torch_dtype=torch_dtype
                    ).to(self.device)

            self._diff_model.scheduler = DDIMScheduler.from_config(
                self._diff_model.scheduler.config
            )

            # Enable memory efficient attention if available
            try:
                self._diff_model.enable_attention_slicing(slice_size="auto")
                print("Enabled attention slicing for memory efficiency")
            except Exception:
                pass

            # Try to enable xFormers or PyTorch 2.0 SDPA
            try:
                self._diff_model.enable_xformers_memory_efficient_attention()
                print("Enabled xFormers memory efficient attention")
            except Exception:
                try:
                    # PyTorch 2.0+ native SDPA
                    from diffusers.models.attention_processor import AttnProcessor2_0
                    self._diff_model.unet.set_attn_processor(AttnProcessor2_0())
                    print("Enabled PyTorch 2.0 SDPA attention")
                except Exception:
                    print("Note: Using standard attention (xFormers/SDPA not available)")

            print(f"Stable Diffusion model loaded successfully (dtype={torch_dtype})")

            # Initialize Adversarial_Opt
            # This loads the FR surrogate models
            self._attacker_model = Adversarial_Opt(self.wdiff_config, self._diff_model)
            
            # Precompute target embeddings
            target_image, _ = get_target_test_images(
                self.wdiff_config.target_choice, self.device, 
                self.wdiff_config.MTCNN_cropping, self.assets_path
            )
            with torch.no_grad():
                 self.target_embeddings = self._attacker_model.get_FR_embeddings(target_image)
            
            print("WeakenDiff (Adversarial_Opt) initialized successfully")
            self._initialized = True

        except Exception as e:
            print(f"Error loading WeakenDiff models: {e}")
            import traceback
            traceback.print_exc()
            self._diff_model = None
            self._initialized = True # Mark as initialized to avoid retrying loop

    def process_frame(self,
                     frame: np.ndarray,
                     face_bbox: Optional[Union[List, Tuple, np.ndarray]] = None,
                     **kwargs) -> np.ndarray:
        """
        Apply WeakenDiff de-identification to a frame.
        """
        # Lazy init
        self._lazy_init()

        if self._diff_model is None or self._attacker_model is None:
            print("Warning: WeakenDiff models not loaded, returning original frame")
            return frame

        try:
            # Prepare image
            # frame is BGR
            # Convert BGR to RGB PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Resize if needed (Adversarial_Opt usually resizes internally or via transforms)
            image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Alignment/Cropping box
            bb_src1 = None
            if self.wdiff_config.MTCNN_cropping:
                if face_bbox is not None:
                     # Use the provided bbox
                     h_orig, w_orig = frame.shape[:2]
                     scale_x = self.wdiff_config.image_size / w_orig
                     scale_y = self.wdiff_config.image_size / h_orig
                     
                     x1, y1, x2, y2 = face_bbox
                     bb_src1 = [
                         int(x1 * scale_x), int(y1 * scale_y),
                         int(x2 * scale_x), int(y2 * scale_y)
                     ]
                else:
                    try:
                        bb_src1 = alignment(pil_image)
                    except:
                        bb_src1 = [0, 0, self.wdiff_config.image_size, self.wdiff_config.image_size]

            # Prepare Controller
            controller = AttentionControlEdit(
                num_steps=self.wdiff_config.diffusion_steps,
                self_replace_steps=1.0
            )

            # Source Embeddings
            source_embeddings = None
            if self.wdiff_config.is_obfuscation:
                image_hold = image_tensor.clone()
                if self.wdiff_config.MTCNN_cropping and bb_src1 is not None:
                     x1, y1, x2, y2 = bb_src1
                     # Clamp
                     x1, y1 = max(0, x1), max(0, y1)
                     x2, y2 = min(self.wdiff_config.image_size, x2), min(self.wdiff_config.image_size, y2)
                     
                     if x2 > x1 and y2 > y1:
                         out_image_hold = image_hold[:, :, y1:y2, x1:x2]
                         image_hold = out_image_hold

                with torch.no_grad():
                    source_embeddings = self._attacker_model.get_FR_embeddings(image_hold)

            # Run Attacker
            latents = self._attacker_model.attacker(
                image_tensor,
                'frame',
                source_embeddings,
                self.target_embeddings,
                controller,
                None, # null_text_dir
                bb_src1
            )
            
            # Decode latent to image (optimized latent is the last one)
            optimized_latent = latents[-1].unsqueeze(0)
            
            with torch.no_grad():
                protected_latent = optimized_latent / 0.18215
                # Ensure latent dtype matches VAE dtype to avoid fp16/fp32 mismatch
                vae_dtype = next(self._diff_model.vae.parameters()).dtype
                protected_latent = protected_latent.to(dtype=vae_dtype)
                image_out = self._diff_model.vae.decode(protected_latent)['sample']
                image_out = (image_out / 2 + 0.5).clamp(0, 1)
                
                protected_np = image_out.cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
                protected_np = (protected_np * 255).astype(np.uint8)

            # Resize back to original size
            h, w = frame.shape[:2]
            protected_np = cv2.resize(protected_np, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Convert RGB to BGR
            result = cv2.cvtColor(protected_np, cv2.COLOR_RGB2BGR)

            return result

        except Exception as e:
            print(f"Error in WeakenDiff processing: {e}")
            import traceback
            traceback.print_exc()
            return frame

    def get_name(self) -> str:
        """Return the name of the de-identification method."""
        return "WeakenDiff"
