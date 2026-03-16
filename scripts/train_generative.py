#!/usr/bin/env python3
"""
Train generative face de-identification models.

This script provides a unified interface for training various generative
de-identification methods: AMT-GAN, Adv-Makeup.

Usage:
    # Train AMT-GAN
    python scripts/train_generative.py --method amtgan --target_path /path/to/target.jpg

    # Train Adv-Makeup
    python scripts/train_generative.py --method advmakeup --targets_dir /path/to/targets/
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config_utils import load_config_into_args


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train generative face de-identification models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Methods:
  - amtgan: AMT-GAN - Adversarial Makeup Transfer GAN
  - advmakeup: Adv-Makeup - Adversarial Makeup Transfer

Examples:
  # Train AMT-GAN with a target identity
  python scripts/train_generative.py --method amtgan \\
      --target_path /path/to/target.jpg \\
      --data_dir /path/to/training/data

  # Train Adv-Makeup with multiple targets
  python scripts/train_generative.py --method advmakeup \\
      --targets_dir /path/to/targets/ \\
      --data_dir /path/to/training/data
"""
    )

    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (CLI args override config values)')

    # Method selection
    parser.add_argument('--method', type=str, required=True,
                        choices=['amtgan', 'advmakeup'],
                        help='Generative method to train')

    # Common training arguments
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to training dataset')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save checkpoints (default: runs/train/method_timestamp)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (default: from config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training (default: from config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: from config)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')

    # AMT-GAN specific arguments
    parser.add_argument('--target_path', type=str, default=None,
                        help='Path to target identity image (for AMT-GAN)')

    # Adv-Makeup specific arguments
    parser.add_argument('--targets_dir', type=str, default=None,
                        help='Directory containing target identity images (for Adv-Makeup)')

    return load_config_into_args(parser)


def train_amtgan(args):
    """Train AMT-GAN model."""
    print("=" * 60)
    print("Training AMT-GAN")
    print("=" * 60)

    # Import AMT-GAN modules
    amtgan_dir = PROJECT_ROOT / 'core' / 'fdeid' / 'generative' / 'amtgan'
    sys.path.insert(0, str(amtgan_dir))

    from torch.backends import cudnn
    from backbone.solver import Solver
    from backbone.config import get_config
    from dataloder import get_loader
    from utils import read_img

    # Get config
    config = get_config()

    # Override with command line arguments
    if args.data_dir is not None:
        config.DATA.PATH = args.data_dir
    if args.epochs is not None:
        config.TRAINING.NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        config.TRAINING.G_LR = args.lr
        config.TRAINING.D_LR = args.lr

    config.DEVICE.device = args.device

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = PROJECT_ROOT / 'runs' / 'train' / f'amtgan_{timestamp}'

    output_dir.mkdir(parents=True, exist_ok=True)
    config.LOG.SNAPSHOT_PATH = str(output_dir / 'checkpoints')
    config.LOG.LOG_PATH = str(output_dir / 'logs')
    config.LOG.VIS_PATH = str(output_dir / 'visualization')

    # Create output directories
    Path(config.LOG.SNAPSHOT_PATH).mkdir(parents=True, exist_ok=True)
    Path(config.LOG.LOG_PATH).mkdir(parents=True, exist_ok=True)
    Path(config.LOG.VIS_PATH).mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {config.DATA.PATH}")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {config.TRAINING.NUM_EPOCHS}")
    print(f"Batch size: {config.DATA.BATCH_SIZE}")
    print(f"Learning rate: {config.TRAINING.G_LR}")
    print("=" * 60)

    # Check target path
    if not args.target_path:
        print("Error: --target_path is required for AMT-GAN training")
        return

    if not os.path.exists(args.target_path):
        print(f"Error: Target image not found: {args.target_path}")
        return

    cudnn.benchmark = True

    # Load data
    data_loader = get_loader(config)

    # Load target image
    target_image = read_img(args.target_path, 0.5, 0.5, config.DEVICE.device)

    # Create solver and train
    solver = Solver(config, target_image, data_loader=data_loader)
    solver.train()

    print("=" * 60)
    print(f"Training completed! Checkpoints saved to: {output_dir}")
    print("=" * 60)


def train_advmakeup(args):
    """Train Adv-Makeup model."""
    print("=" * 60)
    print("Training Adv-Makeup")
    print("=" * 60)

    # Import Adv-Makeup modules
    advmakeup_dir = PROJECT_ROOT / 'core' / 'fdeid' / 'generative' / 'advmakeup'
    sys.path.insert(0, str(advmakeup_dir))

    import torch
    from config import Configuration
    from model import MakeupAttack
    from dataset import dataset_makeup

    # Get config
    config = Configuration()

    # Override with command line arguments
    if args.data_dir is not None:
        config.data_dir = args.data_dir
    if args.epochs is not None:
        config.epoch_steps = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.lr = args.lr

    config.gpu = 0 if args.device == 'cuda' else -1

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = PROJECT_ROOT / 'runs' / 'train' / f'advmakeup_{timestamp}'

    output_dir.mkdir(parents=True, exist_ok=True)
    config.model_dir = str(output_dir / 'checkpoints')
    Path(config.model_dir).mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {config.data_dir}")
    print(f"Targets directory: {args.targets_dir or config.targets_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {config.epoch_steps}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.lr}")
    print("=" * 60)

    device = torch.device(f'cuda:{config.gpu}') if config.gpu >= 0 else torch.device('cpu')

    # Check targets directory
    targets_dir = args.targets_dir or config.targets_dir
    if not os.path.exists(targets_dir):
        print(f"Error: Targets directory not found: {targets_dir}")
        return

    # Training loop for each target
    for target_name in os.listdir(targets_dir):
        if not target_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        print(f"\nTraining for target: {target_name}")

        # Initialize model
        model = MakeupAttack(config)

        # Initialize dataset
        dataset = dataset_makeup(config)
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.n_threads
        )

        # Train
        for ep in range(config.epoch_steps):
            model.res_init(ep)

            for it, data in enumerate(train_loader):
                before = data[0].to(device).detach()
                after = data[1].to(device).detach()
                before_path = data[2]

                # Update discriminator
                model.update_discr(before, after)

                # Update generator
                if (it + 1) % 1 == 0:
                    model.update_gen(before, before_path, target_name)

            # Visualization and logging
            model.visualization(ep, len(train_loader))

            # Save checkpoints
            if (ep + 1) % 50 == 0:
                save_path = os.path.join(config.model_dir, target_name.split('.')[0])
                os.makedirs(save_path, exist_ok=True)
                print(f'Saving model at epoch {ep}')
                torch.save(model.enc.state_dict(), f'{save_path}/{ep:05d}_enc.pth')
                torch.save(model.dec.state_dict(), f'{save_path}/{ep:05d}_dec.pth')
                torch.save(model.discr.state_dict(), f'{save_path}/{ep:05d}_discr.pth')

    print("=" * 60)
    print(f"Training completed! Checkpoints saved to: {output_dir}")
    print("=" * 60)


def main():
    args = parse_args()

    print("=" * 60)
    print("Generative Model Training")
    print("=" * 60)
    print(f"Method: {args.method}")
    print(f"Device: {args.device}")
    print("=" * 60)

    if args.method == 'amtgan':
        train_amtgan(args)
    elif args.method == 'advmakeup':
        train_advmakeup(args)
    else:
        print(f"Error: Unknown method '{args.method}'")
        print("Available methods: amtgan, advmakeup")
        sys.exit(1)


if __name__ == '__main__':
    main()
