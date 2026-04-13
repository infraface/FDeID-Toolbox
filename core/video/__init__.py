"""
Video processing module for face de-identification.

Provides VideoReader, VideoWriter, and VideoProcessor for processing
video files and frame-sequence directories with naive de-identification
methods (blur, pixelation, mask).
"""

from .reader import VideoReader
from .writer import VideoWriter
from .processor import VideoProcessor
from .utils import discover_sources, is_video_file, has_image_files

__all__ = [
    'VideoReader',
    'VideoWriter',
    'VideoProcessor',
    'discover_sources',
    'is_video_file',
    'has_image_files',
]
