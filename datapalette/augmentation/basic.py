# File: datapalette/augmentation/basic.py

import os
import cv2
import numpy as np
import logging
from typing import Optional, List, Tuple
from tqdm import tqdm

logger = logging.getLogger(__name__)

def process_image_batch(input_dir: str, output_dir: str, process_func, batch_size: int = 32, output_format: str = 'png', desc: str = "Processing images"):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for i in tqdm(range(0, len(image_files), batch_size), desc=desc, unit="batch"):
        batch = image_files[i:i+batch_size]
        for img_file in batch:
            img_path = os.path.join(input_dir, img_file)
            img = cv2.imread(img_path)
            processed_img = process_func(img)
            output_filename = f"{os.path.splitext(img_file)[0]}.{output_format}"
            cv2.imwrite(os.path.join(output_dir, output_filename), processed_img)

def rotate_images(input_dir: str, output_dir: Optional[str] = None, angles: List[float] = [90, 180, 270]):
    """Rotate images by specified angles and save in /rotated."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_dir), 'rotated')
    
    logger.info(f"Rotating images in {input_dir} by angles {angles}")
    
    def apply_rotation(img):
        rotated_images = [img]  # Include the original image
        for angle in angles:
            matrix = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1)
            rotated = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
            rotated_images.append(rotated)
        return np.concatenate(rotated_images, axis=1)

    process_image_batch(input_dir, output_dir, apply_rotation, desc="Rotating images")
    logger.info(f"Rotated images saved in {output_dir}")

def mirror_images(input_dir: str, output_dir: Optional[str] = None):
    """Create mirrored versions of images and save in /mirrored."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_dir), 'mirrored')
    
    logger.info(f"Mirroring images in {input_dir}")
    
    def apply_mirroring(img):
        flipped_horizontal = cv2.flip(img, 1)
        flipped_vertical = cv2.flip(img, 0)
        flipped_both = cv2.flip(img, -1)
        return np.concatenate([img, flipped_horizontal, flipped_vertical, flipped_both], axis=1)

    process_image_batch(input_dir, output_dir, apply_mirroring, desc="Mirroring images")
    logger.info(f"Mirrored images saved in {output_dir}")

def adjust_brightness_contrast(input_dir: str, output_dir: Optional[str] = None, 
                               brightness_range: Tuple[float, float] = (0.5, 1.5),
                               contrast_range: Tuple[float, float] = (0.5, 1.5)):
    """Adjust brightness and contrast of images and save in /adjusted."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_dir), 'adjusted')
    
    logger.info(f"Adjusting brightness and contrast of images in {input_dir}")
    
    def apply_adjustment(img):
        brightness = np.random.uniform(*brightness_range)
        contrast = np.random.uniform(*contrast_range)
        adjusted = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
        return np.concatenate([img, adjusted], axis=1)

    process_image_batch(input_dir, output_dir, apply_adjustment, desc="Adjusting brightness and contrast")
    logger.info(f"Adjusted images saved in {output_dir}")

def add_noise(input_dir: str, output_dir: Optional[str] = None, noise_type: str = 'gaussian', amount: float = 0.05):
    """Add noise to images and save in /noisy."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_dir), f'{noise_type}_noise')
    
    logger.info(f"Adding {noise_type} noise to images in {input_dir}")
    
    def apply_noise(img):
        if noise_type == 'gaussian':
            row, col, ch = img.shape
            mean = 0
            sigma = amount * 255
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = img + gauss
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        elif noise_type == 'salt_and_pepper':
            row, col, ch = img.shape
            s_vs_p = 0.5
            noisy = np.copy(img)
            # Salt mode
            num_salt = np.ceil(amount * img.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
            noisy[coords] = 255
            # Pepper mode
            num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
            noisy[coords] = 0
        else:
            logger.error(f"Unsupported noise type: {noise_type}")
            raise ValueError("Unsupported noise type. Choose 'gaussian' or 'salt_and_pepper'.")
        return np.concatenate([img, noisy], axis=1)

    process_image_batch(input_dir, output_dir, apply_noise, desc=f"Adding {noise_type} noise")
    logger.info(f"Noisy images saved in {output_dir}")

def random_crop(input_dir: str, output_dir: Optional[str] = None, crop_size: Tuple[int, int] = (224, 224)):
    """Generate random crops from images and save in /random_crops."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_dir), 'random_crops')
    
    logger.info(f"Generating random crops of size {crop_size} from images in {input_dir}")
    
    def apply_random_crop(img):
        h, w = img.shape[:2]
        top = np.random.randint(0, h - crop_size[0] + 1)
        left = np.random.randint(0, w - crop_size[1] + 1)
        cropped = img[top:top+crop_size[0], left:left+crop_size[1]]
        return cropped

    process_image_batch(input_dir, output_dir, apply_random_crop, desc="Generating random crops")
    logger.info(f"Random crops saved in {output_dir}")

def tile_image(input_dir: str, output_dir: Optional[str] = None, tile_size: Tuple[int, int] = (256, 256), overlap: float = 0.0):
    """
    Tile images into smaller patches with optional overlap.
    
    Args:
    input_dir (str): Directory containing input images.
    output_dir (Optional[str]): Directory to save tiled images. If None, creates a 'tiled' subdirectory in input_dir.
    tile_size (Tuple[int, int]): Size of each tile (width, height).
    overlap (float): Overlap between tiles, as a fraction of tile size (0.0 to 1.0).
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_dir), 'tiled')
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Tiling images in {input_dir} with tile size {tile_size} and overlap {overlap}")
    
    def apply_tiling(img):
        h, w = img.shape[:2]
        tile_h, tile_w = tile_size
        stride_h = int(tile_h * (1 - overlap))
        stride_w = int(tile_w * (1 - overlap))
        
        tiles = []
        for y in range(0, h-tile_h+1, stride_h):
            for x in range(0, w-tile_w+1, stride_w):
                tile = img[y:y+tile_h, x:x+tile_w]
                tiles.append(tile)
        
        return tiles

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_file in tqdm(image_files, desc="Tiling images"):
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        tiles = apply_tiling(img)
        
        base_name = os.path.splitext(img_file)[0]
        for i, tile in enumerate(tiles):
            tile_filename = f"{base_name}_tile_{i}.png"
            cv2.imwrite(os.path.join(output_dir, tile_filename), tile)
    
    logger.info(f"Tiled images saved in {output_dir}")
    
# Example usage:
# rotate_images('input_directory', angles=[45, 90, 135])
# mirror_images('input_directory')
# adjust_brightness_contrast('input_directory', brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2))
# add_noise('input_directory', noise_type='gaussian', amount=0.1)
# random_crop('input_directory', crop_size=(320, 320))