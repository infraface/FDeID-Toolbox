"""AdaFace-ViT (CVLface) wrapper.

Bypasses CVLface's wrapper.py (which depends on OmegaConf and HuggingFace
PreTrainedModel) and loads the underlying ViT model directly via the model dir's
own `models.get_model(...)` factory plus a small DotDict shim.
"""
import importlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml


class DotDict(dict):
    """Dict that also supports attribute access. Recursively wraps nested dicts."""
    def __getattr__(self, key):
        try:
            v = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return DotDict(v) if isinstance(v, dict) else v

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as e:
            raise AttributeError(key) from e


@contextmanager
def _chdir(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


@contextmanager
def _syspath_prepend(path):
    path = str(path)
    inserted = path not in sys.path
    if inserted:
        sys.path.insert(0, path)
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(path)
            except ValueError:
                pass


def _ensure_omegaconf_stub():
    """If real omegaconf is missing, register a minimal stub in sys.modules.

    CVLface's models/base/utils.py does `from omegaconf import DictConfig, OmegaConf`
    at module load time. Only the *save* paths in that file actually call into
    OmegaConf; inference never touches them. Providing a stub lets `import models`
    succeed without installing the real package.
    """
    import sys
    try:
        import omegaconf  # noqa: F401
        return
    except ImportError:
        pass
    import types

    stub = types.ModuleType("omegaconf")

    class DictConfig:  # placeholder used only in isinstance checks elsewhere
        pass

    class _OmegaConf:
        @staticmethod
        def create(*a, **kw):
            raise RuntimeError(
                "omegaconf stub: OmegaConf.create not available; "
                "this code path is only for serialisation, not inference")

        @staticmethod
        def save(*a, **kw):
            raise RuntimeError(
                "omegaconf stub: OmegaConf.save not available; "
                "this code path is only for serialisation, not inference")

        @staticmethod
        def load(*a, **kw):
            raise RuntimeError(
                "omegaconf stub: OmegaConf.load not available; "
                "this code path is only for serialisation, not inference")

    stub.DictConfig = DictConfig
    stub.OmegaConf = _OmegaConf
    sys.modules["omegaconf"] = stub


def _ensure_fvcore_stub():
    """If real fvcore is missing, register a minimal stub in sys.modules.

    CVLface's models/vit/vit.py does `from fvcore.nn import flop_count` at
    module load time. flop_count is only used for parameter profiling, never
    during inference. A no-op stub lets the import succeed.
    """
    import sys
    try:
        import fvcore  # noqa: F401
        return
    except ImportError:
        pass
    import types

    fvcore_mod = types.ModuleType("fvcore")
    nn_mod = types.ModuleType("fvcore.nn")

    def flop_count(*args, **kwargs):
        raise RuntimeError(
            "fvcore stub: flop_count not available; "
            "this code path is only for profiling, not inference")

    nn_mod.flop_count = flop_count
    fvcore_mod.nn = nn_mod
    sys.modules["fvcore"] = fvcore_mod
    sys.modules["fvcore.nn"] = nn_mod


def _load_model_from_dir(model_dir: Path):
    """Import the model dir's own `models` package (its __init__ defines get_model)."""
    _ensure_omegaconf_stub()
    _ensure_fvcore_stub()
    # Ensure any stale 'models' import from a previous call doesn't shadow.
    for cached in [k for k in list(sys.modules) if k == "models" or k.startswith("models.")]:
        del sys.modules[cached]

    with _syspath_prepend(model_dir), _chdir(model_dir):
        models_pkg = importlib.import_module("models")
        cfg_dict = yaml.safe_load(open(model_dir / "pretrained_model" / "model.yaml"))
        conf = DotDict(cfg_dict)
        model = models_pkg.get_model(conf)
        # The model exposes load_state_dict_from_path per CVLface convention.
        model.load_state_dict_from_path("pretrained_model/model.pt")
    return model


class AdaFaceViT:
    def __init__(self, model_dir: str, device: str = "cuda"):
        self.device = device
        self.model_dir = Path(model_dir).resolve()
        self.model = _load_model_from_dir(self.model_dir)
        self.model.eval().to(device)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, 112, 112] in the model's expected color space.
        Returns L2-normalised [B, output_dim] (typically 512)."""
        x = x.to(self.device)
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return F.normalize(out, p=2, dim=1)
