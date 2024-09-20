# File: datapalette/pipelines/predefined.py

import os
import logging
from typing import List, Callable
from tqdm import tqdm
from ..core.functions import extract_frames, n_croppings_from_frame
from ..preprocessing.advanced import (
    pca_color_augmentation, convert_to_hsv_and_combine, convert_to_lab_and_combine,
    fourier_transform_channels, add_gradient_channels, convert_to_multispectral,
    add_edge_channels, enhance_green_channel
)
from ..augmentation.basic import (
    rotate_images, mirror_images, adjust_brightness_contrast, add_noise, random_crop
)

logger = logging.getLogger(__name__)

def gan_pipeline(input_path: str, output_dir: str, fps: int = 1, crop_size: tuple = (256, 256), num_crops: int = 5):
    """Pipeline for preparing data for GANs."""
    logger.info(f"Starting GAN pipeline processing for {input_path}")
    
    # Step 1: Extract frames from video
    logger.info("Step 1: Extracting frames from video")
    frames_dir = os.path.join(output_dir, 'video_frames')
    extract_frames(os.path.dirname(input_path), os.path.basename(input_path), frames_dir, fps)
    
    # Step 2: Create multiple random crops
    logger.info("Step 2: Creating multiple random crops")
    crops_dir = os.path.join(output_dir, 'cropped_frames')
    for frame in tqdm(os.listdir(frames_dir), desc="Cropping frames"):
        n_croppings_from_frame(os.path.join(frames_dir, frame), crops_dir, num_crops, crop_size)
    
    # Step 3: Apply basic augmentations
    logger.info("Step 3: Applying basic augmentations")
    rotate_images(crops_dir, os.path.join(output_dir, 'rotated'))
    mirror_images(crops_dir, os.path.join(output_dir, 'mirrored'))
    
    # Step 4: Apply advanced preprocessing
    logger.info("Step 4: Applying advanced preprocessing")
    pca_color_augmentation(crops_dir, os.path.join(output_dir, 'pca_augmented'))
    adjust_brightness_contrast(crops_dir, os.path.join(output_dir, 'adjusted'))
    
    logger.info("GAN pipeline processing completed")

def unet_segmentation_pipeline(input_dir: str, output_dir: str):
    """Pipeline for preparing data for image segmentation with U-Net."""
    logger.info(f"Starting U-Net segmentation pipeline processing for {input_dir}")
    
    # Step 1: Convert to different color spaces
    logger.info("Step 1: Converting to different color spaces")
    convert_to_hsv_and_combine(input_dir, os.path.join(output_dir, 'hsv_combined'))
    convert_to_lab_and_combine(input_dir, os.path.join(output_dir, 'lab_combined'))
    
    # Step 2: Add edge and gradient information
    logger.info("Step 2: Adding edge and gradient information")
    add_gradient_channels(input_dir, os.path.join(output_dir, 'gradient_channels'))
    add_edge_channels(input_dir, os.path.join(output_dir, 'edge_channels'))
    
    # Step 3: Enhance green channel (useful for medical imaging)
    logger.info("Step 3: Enhancing green channel")
    enhance_green_channel(input_dir, os.path.join(output_dir, 'green_enhanced'))
    
    # Step 4: Apply basic augmentations
    logger.info("Step 4: Applying basic augmentations")
    rotate_images(input_dir, os.path.join(output_dir, 'rotated'))
    mirror_images(input_dir, os.path.join(output_dir, 'mirrored'))
    
    logger.info("U-Net segmentation pipeline processing completed")

def diffusion_model_pipeline(input_dir: str, output_dir: str):
    """Pipeline for preparing data for diffusion models."""
    logger.info(f"Starting diffusion model pipeline processing for {input_dir}")
    
    # Step 1: Apply color augmentation
    logger.info("Step 1: Applying color augmentation")
    pca_color_augmentation(input_dir, os.path.join(output_dir, 'pca_augmented'))
    
    # Step 2: Convert to multispectral
    logger.info("Step 2: Converting to multispectral")
    convert_to_multispectral(input_dir, os.path.join(output_dir, 'multispectral'))
    
    # Step 3: Apply Fourier transform
    logger.info("Step 3: Applying Fourier transform")
    fourier_transform_channels(input_dir, os.path.join(output_dir, 'fourier_transformed'))
    
    # Step 4: Add noise (simulating different noise levels)
    logger.info("Step 4: Adding noise at different levels")
    add_noise(input_dir, os.path.join(output_dir, 'noisy_low'), noise_type='gaussian', amount=0.01)
    add_noise(input_dir, os.path.join(output_dir, 'noisy_medium'), noise_type='gaussian', amount=0.05)
    add_noise(input_dir, os.path.join(output_dir, 'noisy_high'), noise_type='gaussian', amount=0.1)
    
    # Step 5: Apply random cropping
    logger.info("Step 5: Applying random cropping")
    random_crop(input_dir, os.path.join(output_dir, 'random_crops'), crop_size=(224, 224))
    
    logger.info("Diffusion model pipeline processing completed")

def custom_pipeline(input_path: str, output_dir: str, steps: List[Callable]):
    """Apply a custom pipeline of processing steps."""
    logger.info(f"Starting custom pipeline processing for {input_path}")
    current_input = input_path
    for i, step in enumerate(tqdm(steps, desc="Executing pipeline steps")):
        logger.info(f"Executing step {i+1}")
        step_output_dir = os.path.join(output_dir, f'step_{i+1}')
        step(current_input, step_output_dir)
        current_input = step_output_dir
    logger.info("Custom pipeline processing completed")

# Example usage of custom pipeline
# custom_steps = [
#     lambda x, y: rotate_images(x, y, angles=[45, 135]),
#     lambda x, y: pca_color_augmentation(x, y, alpha_std=0.05),
#     lambda x, y: add_gradient_channels(x, y),
# ]
# custom_pipeline('input_dir', 'output_dir', custom_steps)