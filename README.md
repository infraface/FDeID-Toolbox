<img src="./assets/logo.png" width=100%>

# Face De-identification Toolbox

A modular face de-identification toolbox for privacy-preserving facial analysis research.

## Quick Start

### Environment Setup
**FDeID-Toolbox** is built exclusively on PyTorch, ensuring a lightweight and unified environment. No complex C++ extensions or conflicting frameworks.
```bash
# 1. Clone the repo
git clone https://github.com/infraface/FDeID-Toolbox.git
cd FDeID-Toolbox

# 2. Install dependencies (Only PyTorch and standard vision libs)
pip install -r requirements.txt
```

### Running an Experiment

Every experiment is fully specified by a single YAML config file. CLI arguments can override any config value.

```bash
# Run from config only
python scripts/run_naive_deid.py --config configs/naive/blur_lfw.yaml

# Override specific parameters
python scripts/run_naive_deid.py --config configs/naive/blur_lfw.yaml --kernel_size 100 --max_images 50

# Run without config (traditional CLI usage still works)
python scripts/run_naive_deid.py --dataset lfw --method blur --save_dir runs/inference/blur_lfw
```

### Running Evaluation

```bash
python scripts/eval_privacy_lfw.py --config configs/eval/privacy_lfw.yaml --deid_dir runs/inference/blur_lfw
python scripts/eval_quality.py --config configs/eval/quality.yaml --original_dir /path/to/lfw --deid_dir runs/inference/blur_lfw
```

## YAML Configuration

All scripts accept `--config path/to/config.yaml`. The YAML keys correspond directly to CLI argument names (without the `--` prefix). CLI arguments always override YAML values.

**Inference config example** (`configs/naive/blur_lfw.yaml`):
```yaml
dataset: lfw
method: blur
kernel_size: 60
save_dir: runs/inference/blur_lfw
device: cuda
```

**Evaluation config example** (`configs/eval/privacy_lfw.yaml`):
```yaml
original_dir: /path/to/datasets/Dataset_LFW
deid_dir: null  # Provide via CLI: --deid_dir /path/to/deid
output_dir: runs/eval/privacy_lfw
device: cuda
models:
  - arcface
  - cosface
  - adaface
```

## BaseDeIdentifier Class

All de-identification methods inherit from `BaseDeIdentifier` (`core/fdeid/base.py`), providing a unified interface.

### Interface

```python
from core.fdeid.base import BaseDeIdentifier

class BaseDeIdentifier(ABC):
    def __init__(self, config: dict):
        """Initialize with a configuration dictionary."""
        self.config = config
        self.device = config.get('device', 'cuda')

    @abstractmethod
    def process_frame(self, frame: np.ndarray, face_bbox=None, **kwargs) -> np.ndarray:
        """
        Apply de-identification to a single frame.

        Args:
            frame: Input image (H, W, C) in BGR format, range [0, 255]
            face_bbox: Optional bounding box (x1, y1, x2, y2)

        Returns:
            De-identified frame (H, W, C)
        """
        pass

    def process_batch(self, frames: torch.Tensor, face_bboxes=None, **kwargs) -> torch.Tensor:
        """Process a batch of frames (B, C, H, W). Override for native batch support."""

    def get_name(self) -> str:
        """Return method name."""

    def get_config(self) -> dict:
        """Return configuration dictionary."""

    def to(self, device):
        """Move to device."""
```

### Usage Examples

#### Direct instantiation

```python
from core.fdeid.naive import GaussianBlurDeIdentifier

config = {'kernel_size': 60, 'device': 'cuda'}
deid = GaussianBlurDeIdentifier(config)
result = deid.process_frame(image, face_bbox=[x1, y1, x2, y2])
```

#### Factory function

```python
from core.fdeid import get_deidentifier

# Naive method
config = {'type': 'naive', 'method_name': 'blur', 'kernel_size': 60, 'device': 'cuda'}
deid = get_deidentifier(config)
result = deid.process_frame(image, face_bbox=[x1, y1, x2, y2])

# Generative method
config = {'type': 'generative', 'method_name': 'ciagan', 'device': 'cuda'}
deid = get_deidentifier(config)
result = deid.process_frame(image, face_bbox=[x1, y1, x2, y2])
```

#### Implementing a new method

```python
import numpy as np
from core.fdeid.base import BaseDeIdentifier

class MyDeIdentifier(BaseDeIdentifier):
    def __init__(self, config):
        super().__init__(config)
        self.intensity = config.get('intensity', 1.0)

    def process_frame(self, frame, face_bbox=None, **kwargs):
        if face_bbox is None:
            return frame

        result = frame.copy()
        x1, y1, x2, y2 = [int(c) for c in face_bbox]

        # Apply your de-identification to the face region
        face = result[y1:y2, x1:x2]
        face = self._transform(face)
        result[y1:y2, x1:x2] = face

        return result

    def _transform(self, face):
        # Your custom transformation
        return (face * self.intensity).clip(0, 255).astype(np.uint8)
```

To integrate into the factory system, register in `core/fdeid/__init__.py`.

## Available Methods

| Category | Method | Config Key |
|----------|--------|------------|
| Naive | Gaussian Blur | `blur` |
| Naive | Pixelation | `pixelate` |
| Naive | Black Mask | `mask` |
| Generative | CIAGAN | `ciagan` |
| Generative | AMT-GAN | `amtgan` |
| Generative | Adv-Makeup | `advmakeup` |
| Generative | WeakenDiff | `weakendiff` |
| Generative | DeID-rPPG | `deid_rppg` |
| Generative | G2Face | `g2face` |
| Adversarial | PGD | `pgd` |
| Adversarial | MI-FGSM | `mifgsm` |
| Adversarial | TI-DIM | `tidim` |
| Adversarial | TI-PIM | `tipim` |
| Adversarial | Chameleon | `chameleon` |
| K-Same | k-Same-Average | `average` |
| K-Same | k-Same-Select | `select` |
| K-Same | k-Same-Furthest | `furthest` |

## Evaluation Metrics

| Category | Metric | Script |
|----------|--------|--------|
| Privacy | Verification accuracy, TAR@FAR, PSR | `eval_privacy_lfw.py`, `eval_privacy_agedb.py` |
| Quality | PSNR, SSIM, LPIPS, FID, NIQE | `eval_quality.py` |
| Utility - Age | MAE | `eval_age.py` |
| Utility - Gender | Accuracy | `eval_gender.py` |
| Utility - Expression | Accuracy | `eval_expression.py` |
| Utility - Landmark | NME | `eval_landmark.py` |
| Utility - Ethnicity | Accuracy | `eval_ethnicity.py` |
| Utility - rPPG | Heart rate MAE, RMSE | `eval_rppg_utility.py` |

## Supported Datasets

| Dataset | Path |
|---------|------|
| LFW | `/path/to/datasets/Dataset_LFW` |
| AgeDB | `/path/to/datasets/AgeDB` |
| AffectNet | `/path/to/datasets/AffectNet` |
| CelebA-HQ | `/path/to/datasets/Dataset_CelebA_HQ/celeba_hq` |
| FairFace | `/path/to/datasets/FairFace` |
| PURE | `/path/to/datasets/PURE` |

## Acknowledgments

This project would not have been possible without the open-source community.

We thank the authors and maintainers of the repositories and pretrained models that our toolbox builds on, reimplements, or interfaces with. We acknowledge work related to:

- RetinaFace and Dlib for face detection and landmark localization
- ArcFace, CosFace, and AdaFace for face recognition
- FairFace, POSTER, HRNet, and FactorizePhys for utility evaluation
- k-Same-Average, k-Same-Select, and k-Same-Furthest
- PGD, MI-FGSM, TI-DIM, TIP-IM, and Chameleon
- CIAGAN, AMT-GAN, Adv-Makeup, WeakenDiff, DeID-rPPG, and G$^{2}$Face

We also thank the creators of datasets such as LFW and AgeDB.

Please support these original projects by citing their papers and visiting their repositories.