"""
Video writer supporting both video files and frame-sequence directories.

Provides a unified interface for writing frames to video files
(via cv2.VideoWriter) or directories of image frames.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple


class VideoWriter:
    """
    Unified video writer for video files and frame directories.

    Usage:
        # Write to video file
        with VideoWriter('/path/to/output.mp4', fps=30.0, frame_size=(640, 480)) as writer:
            writer.write(frame)

        # Write to frame directory (preserving original filenames)
        with VideoWriter('/path/to/output_dir', frame_names=['001.png', '002.png']) as writer:
            writer.write(frame)
    """

    def __init__(self,
                 path: str,
                 fps: float = 30.0,
                 frame_size: Optional[Tuple[int, int]] = None,
                 codec: str = 'mp4v',
                 frame_names: Optional[List[str]] = None):
        """
        Args:
            path: Output video file path or directory for frame images.
            fps: Frames per second (for video output).
            frame_size: (width, height) tuple (for video output).
            codec: FourCC codec string (default 'mp4v').
            frame_names: Original filenames to use for frame directory output.
                         If None, frames are numbered sequentially.
        """
        self._path = Path(path)
        self._fps = fps
        self._frame_size = frame_size
        self._codec = codec
        self._frame_names = frame_names
        self._writer = None
        self._frame_idx = 0
        self._is_video = False

        # Determine mode based on frame_names or file extension
        if frame_names is not None:
            self._is_video = False
            self._path.mkdir(parents=True, exist_ok=True)
        elif self._path.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}:
            self._is_video = True
            self._path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Default to frame directory
            self._is_video = False
            self._path.mkdir(parents=True, exist_ok=True)

    def _init_video_writer(self, frame: np.ndarray):
        """Lazily initialize cv2.VideoWriter on first frame."""
        h, w = frame.shape[:2]
        if self._frame_size is None:
            self._frame_size = (w, h)
        fourcc = cv2.VideoWriter_fourcc(*self._codec)
        self._writer = cv2.VideoWriter(
            str(self._path), fourcc, self._fps, self._frame_size
        )
        if not self._writer.isOpened():
            raise IOError(f"Cannot open video writer: {self._path}")

    def write(self, frame: np.ndarray):
        """Write a single frame."""
        if self._is_video:
            if self._writer is None:
                self._init_video_writer(frame)
            self._writer.write(frame)
        else:
            if self._frame_names and self._frame_idx < len(self._frame_names):
                fname = self._frame_names[self._frame_idx]
            else:
                fname = f"{self._frame_idx:06d}.png"
            out_path = self._path / fname
            cv2.imwrite(str(out_path), frame)
        self._frame_idx += 1

    @property
    def frames_written(self) -> int:
        return self._frame_idx

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def release(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
