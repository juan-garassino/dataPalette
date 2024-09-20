# File: datapalette/core/functions.py

import os
import cv2
import numpy as np
import logging
from typing import Optional, Tuple
from tqdm import tqdm

logger = logging.getLogger(__name__)

def extract_frames(input_dir: str, input_file: str, output_dir: str, fps: int = 1, output_size: Optional[Tuple[int, int]] = None):
    """Extract frames from a video and save them in a /frames folder."""
    video_path = os.path.join(input_dir, input_file)
    #frames_dir = os.path.join(output_dir)#, 'frames')
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Extracting frames from {input_file}")
    cap = cv2.VideoCapture(video_path)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, int(video_fps / fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    saved_count = 0

    with tqdm(total=total_frames, desc="Extracting frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                if output_size:
                    frame = cv2.resize(frame, output_size)
                
                frame_filename = f"frame_{saved_count:06d}.png"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                saved_count += 1

            frame_count += 1
            pbar.update(1)

    cap.release()
    logger.info(f"Extracted {saved_count} frames from {input_file}")

def n_croppings_from_frame(input_frame: str, output_dir: str, num_crops: int, crop_size: Tuple[int, int]):
    """Generate N random crops from a resized frame and save them in a /cropped_frames folder."""
    #cropped_frames_dir = os.path.join(output_dir, 'cropped_frames')
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Generating {num_crops} crops from {input_frame}")
    frame = cv2.imread(input_frame)
    if frame is None:
        logger.error(f"Could not read image file: {input_frame}")
        raise ValueError(f"Could not read image file: {input_frame}")

    h, w = frame.shape[:2]

    for i in tqdm(range(num_crops), desc="Generating crops", unit="crop"):
        # Ensure that the crop doesn't exceed the image boundaries
        max_x = max(0, w - crop_size[0])
        max_y = max(0, h - crop_size[1])
        
        x = np.random.randint(0, max_x + 1)
        y = np.random.randint(0, max_y + 1)

        crop = frame[y:y+crop_size[1], x:x+crop_size[0]]

        # Ensure that the crop has the correct size (in case of boundary issues)
        if crop.shape[:2] != crop_size:
            crop = cv2.resize(crop, crop_size)

        crop_filename = f"crop_{i:03d}_{os.path.basename(input_frame)}"
        crop_path = os.path.join(output_dir, crop_filename)
        cv2.imwrite(crop_path, crop)

    logger.info(f"Generated {num_crops} crops from {input_frame}")

# Example usage:
# extract_frames("input_videos", "example.mp4", "output_directory", fps=2, output_size=(640, 480))
# n_croppings_from_frame("output_directory/frames/frame_000001.png", "output_directory", num_crops=5, crop_size=(224, 224))