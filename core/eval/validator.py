"""
Validator Module

This module defines the BaseValidator class, which serves as a foundation for validating
various models including face recognition, age estimation, gender recognition,
ethnicity recognition, and emotion recognition models.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


class BaseValidator(ABC):
    """
    Base class for validating models on a test or validation split of a dataset.

    Supported tasks:
    - Face Recognition
    - Age Estimation
    - Gender Recognition
    - Ethnicity Recognition
    - Emotion Recognition
    """

    def __init__(self, model: Any, device: str = "cuda:0"):
        """
        Initialize the validator.

        Args:
            model (Any): The model to be validated.
            device (str): The device to run validation on (e.g., 'cuda:0', 'cpu').
        """
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    @abstractmethod
    def get_dataloader(self, dataloader, batch_size: int = 32, num_workers: int = 4) -> DataLoader:
        """
        Get the dataloader for the dataset.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Batch size for the dataloader.
            num_workers (int): Number of workers for data loading.

        Returns:
            DataLoader: The PyTorch DataLoader.
        """
        pass

    @abstractmethod
    def preprocess(self, batch: Any) -> Tuple[Any, Any]:
        """
        Preprocess a batch of data.

        Args:
            batch (Any): A batch of data from the dataloader.

        Returns:
            Tuple[Any, Any]: A tuple containing (inputs, targets).
                             Inputs should be ready to be passed to the model (except device transfer).
        """
        pass

    @abstractmethod
    def init_metrics(self) -> None:
        """
        Initialize metric trackers.
        """
        pass

    @abstractmethod
    def update_metrics(self, outputs: Any, targets: Any) -> None:
        """
        Update metrics with the results of the current batch.

        Args:
            outputs (Any): The outputs from the model.
            targets (Any): The ground truth targets.
        """
        pass

    @abstractmethod
    def finalize_metrics(self) -> Dict[str, float]:
        """
        Compute and return the final metrics.

        Returns:
            Dict[str, float]: A dictionary mapping metric names to their values.
        """
        pass

    def plot_predictions(self, inputs: Any, outputs: Any, targets: Any, save_path: Optional[str] = None) -> None:
        """
        Plot predictions for visualization.

        Args:
            inputs (Any): The inputs to the model.
            outputs (Any): The outputs from the model.
            targets (Any): The ground truth targets.
            save_path (Optional[str]): Path to save the plot.
        """
        pass

    def __call__(self, dataloader, batch_size: int = 32, num_workers: int = 4) -> Dict[str, float]:
        """
        Run the validation loop.

        Args:
            dataloader.
            batch_size (int): Batch size.
            num_workers (int): Number of workers.

        Returns:
            Dict[str, float]: The final metrics.
        """
        from tqdm import tqdm
        self.init_metrics()

        # Use tqdm for progress bar if available
        # pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Validation")

        with torch.no_grad():
            # for i, batch in pbar:
            for i, batch in enumerate(dataloader):
                inputs, targets = self.preprocess(batch)

                # Skip empty batches (can happen in distributed training)
                if isinstance(inputs, torch.Tensor):
                    if inputs.numel() == 0:
                        continue
                elif isinstance(inputs, (list, tuple)):
                    if len(inputs) == 0 or (isinstance(inputs[0], torch.Tensor) and inputs[0].numel() == 0):
                        continue

                # Move inputs to device
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(self.device)
                elif isinstance(inputs, (list, tuple)):
                    inputs = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in inputs]
                elif isinstance(inputs, dict):
                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                # Forward pass
                outputs = self.model(inputs)

                # Update metrics
                self.update_metrics(outputs, targets)

        return self.finalize_metrics()
