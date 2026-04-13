"""
Video processing utility functions.

Provides constants and helpers for discovering video files and
frame-sequence directories.
"""

import os
from pathlib import Path
from typing import List, Tuple


VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp'}


def is_video_file(path: str) -> bool:
    """Check if a path points to a video file based on extension."""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def has_image_files(dir_path: str) -> bool:
    """Check if a directory contains image files."""
    dp = Path(dir_path)
    if not dp.is_dir():
        return False
    for f in dp.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            return True
    return False


def discover_sources(input_path: str, input_type: str = 'auto') -> List[Tuple[str, str]]:
    """
    Discover video sources from an input path.

    Args:
        input_path: Path to a video file, directory of videos, or
                    directory containing frame-sequence subdirectories.
        input_type: One of 'auto', 'video', 'frame_sequence'.

    Returns:
        List of (source_path, source_type) tuples where source_type is
        'video' or 'frame_sequence'.
    """
    p = Path(input_path)

    if input_type == 'video':
        if p.is_file() and is_video_file(str(p)):
            return [(str(p), 'video')]
        elif p.is_dir():
            sources = []
            for f in sorted(p.iterdir()):
                if f.is_file() and is_video_file(str(f)):
                    sources.append((str(f), 'video'))
            return sources
        return []

    if input_type == 'frame_sequence':
        if p.is_dir() and has_image_files(str(p)):
            return [(str(p), 'frame_sequence')]
        elif p.is_dir():
            sources = []
            for d in sorted(p.iterdir()):
                if d.is_dir() and has_image_files(str(d)):
                    sources.append((str(d), 'frame_sequence'))
            return sources
        return []

    # Auto-detect
    if p.is_file() and is_video_file(str(p)):
        return [(str(p), 'video')]

    if p.is_dir():
        # Check if the directory itself is a frame sequence
        if has_image_files(str(p)):
            return [(str(p), 'frame_sequence')]

        # Check for video files in this directory
        video_files = sorted([
            f for f in p.iterdir()
            if f.is_file() and is_video_file(str(f))
        ])
        if video_files:
            return [(str(f), 'video') for f in video_files]

        # Check for subdirectories that are frame sequences
        frame_dirs = sorted([
            d for d in p.iterdir()
            if d.is_dir() and has_image_files(str(d))
        ])
        if frame_dirs:
            return [(str(d), 'frame_sequence') for d in frame_dirs]

    return []
