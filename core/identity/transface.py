"""TransFace-B face recognition wrapper.

Matches the encode() surface area used by ArcFace/CosFace/AdaFace in this package.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Project root on sys.path so 'from core.identity.transface_repo import ...' works.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.identity.transface_repo import get_model  # noqa: E402


class TransFace:
    def __init__(self, weight_path: str, network: str = "vit_b",
                 device: str = "cuda", fp16: bool = False):
        self.device = device
        self.model = get_model(network, fp16=fp16)
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Strip "module." prefix if present (DDP checkpoints)
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[TransFace] missing keys ({len(missing)}): "
                  f"{missing[:3]}{'...' if len(missing) > 3 else ''}")
        if unexpected:
            print(f"[TransFace] unexpected keys ({len(unexpected)}): "
                  f"{unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")
        self.model.eval().to(device)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,3,112,112] in [-1,1]. Returns L2-normalised [B,512]."""
        x = x.to(self.device)
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return F.normalize(out, p=2, dim=1)
