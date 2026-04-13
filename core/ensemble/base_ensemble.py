"""
Base class for ensemble face de-identification methods.

Provides shared infrastructure for building and managing
multiple child de-identification methods.
"""

from typing import Dict, Any, List
import torch

from core.fdeid.base import BaseDeIdentifier
from core.fdeid import get_deidentifier


class BaseEnsembleDeIdentifier(BaseDeIdentifier):
    """
    Base class for all ensemble de-identification strategies.

    Manages a list of child de-identification methods built from
    individual method configs. Subclasses implement specific
    combination strategies (sequential, parallel, attribute-guided).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.methods: List[BaseDeIdentifier] = []
        method_configs = config.get('methods', [])
        self._build_methods(method_configs)

    def _build_methods(self, method_configs: List[Dict[str, Any]]):
        """
        Build child de-identifiers from a list of method configs.

        Each config dict should have 'type' and 'method_name' keys
        matching the core.fdeid factory interface.
        """
        for cfg in method_configs:
            cfg.setdefault('device', str(self.device))
            method = get_deidentifier(cfg)
            self.methods.append(method)

    def to(self, device: torch.device):
        """Move all child methods to the specified device."""
        super().to(device)
        for m in self.methods:
            m.to(device)
        return self

    def get_name(self) -> str:
        """Return a descriptive name including child method names."""
        child_names = [m.get_name() for m in self.methods]
        return f"{self.__class__.__name__}({', '.join(child_names)})"

    def get_config(self) -> Dict[str, Any]:
        """Return config including child method configs."""
        cfg = self.config.copy()
        cfg['child_configs'] = [m.get_config() for m in self.methods]
        return cfg
