# File: datapalette/main.py

import argparse
import os
import logging
import cv2
import yaml
from tqdm import tqdm
from datapalette.pipelines.predefined import gan_pipeline, unet_segmentation_pipeline, diffusion_model_pipeline
from datapalette.utils.helpers import setup_logging
from datapalette.preprocessing.advanced import *
from datapalette.augmentation.basic import *
from datapalette.core.functions import extract_frames

logger = logging.getLogger(__name__)

# Set default directories
DEFAULT_INPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def showcase_pipeline(input_image: str, output_dir: str, steps: list):
    """Process a single image through all steps, saving the output of each step."""
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(input_image)
    
    for i, (step_name, step_func) in enumerate(steps):
        logger.info(f"Applying step {i+1}/{len(steps)}: {step_name}")
        img = step_func(img)
        output_path = os.path.join(output_dir, f"{i+1:02d}_{step_name}.png")
        cv2.imwrite(output_path, img)
        logger.info(f"Saved output of step '{step_name}' to {output_path}")

def get_pipeline_steps(args):
    """Define pipeline steps based on provided arguments."""
    steps = []
    if args.rotate:
        steps.append(('Rotation', lambda img: rotate_images(img, angles=args.rotate_angles)))
    if args.mirror:
        steps.append(('Mirroring', mirror_images))
    if args.brightness_contrast:
        steps.append(('Brightness/Contrast Adjustment', lambda img: adjust_brightness_contrast(img, brightness_range=args.brightness_range, contrast_range=args.contrast_range)))
    if args.noise:
        steps.append(('Noise Addition', lambda img: add_noise(img, noise_type=args.noise_type, amount=args.noise_amount)))
    if args.crop:
        steps.append(('Random Cropping', lambda img: random_crop(img, crop_size=tuple(args.crop_size))))
    if args.pca:
        steps.append(('PCA Color Augmentation', lambda img: pca_color_augmentation(img, alpha_std=args.pca_alpha_std)))
    if args.hsv:
        steps.append(('HSV Conversion', convert_to_hsv_and_combine))
    if args.lab:
        steps.append(('LAB Conversion', convert_to_lab_and_combine))
    if args.fourier:
        steps.append(('Fourier Transform', fourier_transform_channels))
    if args.gradient:
        steps.append(('Gradient Channels', add_gradient_channels))
    if args.multispectral:
        steps.append(('Multispectral Conversion', convert_to_multispectral))
    if args.edge:
        steps.append(('Edge Channels', add_edge_channels))
    if args.green:
        steps.append(('Green Channel Enhancement', enhance_green_channel))
    if args.custom_filter:
        steps.append(('Custom Filter', lambda img: create_custom_filter_channels(img, filter_type=args.custom_filter)))
    return steps

def process_input(args):
    """Process input based on input type."""
    if args.input_type == 'video':
        if not os.path.isfile(args.input):
            raise ValueError("For video input, please provide a path to a video file")
        video_frames_dir = os.path.join(args.output_dir, 'video_frames')
        os.makedirs(video_frames_dir, exist_ok=True)
        logger.info(f"Extracting frames from video: {args.input}")
        extract_frames(os.path.dirname(args.input), os.path.basename(args.input), video_frames_dir, fps=args.fps)
        return video_frames_dir
    elif args.input_type == 'images':
        if not os.path.isdir(args.input):
            raise ValueError("For image input, please provide a path to a directory of images")
        return args.input
    elif args.input_type == 'single_image':
        if not os.path.isfile(args.input):
            raise ValueError("For single image input, please provide a path to an image file")
        return args.input
    else:
        raise ValueError(f"Invalid input type: {args.input_type}")

def apply_tiling(input_path, args):
    """Apply tiling if specified."""
    if args.tile and args.pipeline != 'showcase':
        tiled_dir = os.path.join(args.output_dir, 'tiled_input')
        logger.info(f"Applying tiling to input images")
        tile_image(input_path, tiled_dir, tuple(args.tile_size), args.tile_overlap)
        return tiled_dir
    return input_path

def execute_pipeline(args, input_path):
    """Execute the specified pipeline."""
    if args.pipeline == 'gan':
        if args.input_type != 'video':
            raise ValueError("GAN pipeline requires video input")
        logger.info("Executing GAN pipeline")
        gan_pipeline(input_path, args.output_dir, crop_size=tuple(args.crop_size), num_crops=args.num_crops)
    elif args.pipeline == 'unet':
        logger.info("Executing U-Net segmentation pipeline")
        unet_segmentation_pipeline(input_path, args.output_dir)
    elif args.pipeline == 'diffusion':
        logger.info("Executing diffusion model pipeline")
        diffusion_model_pipeline(input_path, args.output_dir)
    elif args.pipeline in ['custom', 'showcase']:
        steps = get_pipeline_steps(args)
        if args.pipeline == 'showcase':
            showcase_pipeline(input_path, args.output_dir, steps)
        else:
            for i, (step_name, step_func) in enumerate(tqdm(steps, desc="Executing custom pipeline")):
                logger.info(f"Executing step {i+1}/{len(steps)}: {step_name}")
                step_output_dir = os.path.join(args.output_dir, f'step_{i+1}_{step_name.lower().replace("/", "_")}')
                step_func(input_path if i == 0 else step_output_dir, step_output_dir)
                input_path = step_output_dir
    else:
        raise ValueError(f"Invalid pipeline: {args.pipeline}")

def main():
    parser = argparse.ArgumentParser(description="DataPalette: Image Dataset Creation and Preprocessing")
    parser.add_argument("--input", default=DEFAULT_INPUT_DIR, help="Input directory of images, video file, or single image for showcase")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for processed images")
    parser.add_argument("--pipeline", choices=['gan', 'unet', 'diffusion', 'custom', 'showcase'], required=True, help="Pipeline to use")
    parser.add_argument("--input-type", choices=['video', 'images', 'single_image'], required=True, help="Type of input")
    parser.add_argument("--config", help="Path to configuration file")
    
    # Video-specific arguments
    parser.add_argument("--fps", type=int, default=1, help="Frames per second to extract (for video input)")
    
    # GAN pipeline specific arguments
    parser.add_argument("--crop-size", nargs=2, type=int, default=[256, 256], help="Crop size for GAN pipeline")
    parser.add_argument("--num-crops", type=int, default=5, help="Number of crops per frame (for GAN pipeline)")
    
    # Tiling arguments
    parser.add_argument("--tile", action="store_true", help="Apply tiling to input images")
    parser.add_argument("--tile-size", nargs=2, type=int, default=[256, 256], help="Size of tiles (width height)")
    parser.add_argument("--tile-overlap", type=float, default=0.0, help="Overlap between tiles (0.0 to 1.0)")
    
    # Custom pipeline arguments
    parser.add_argument("--rotate", action="store_true", help="Apply rotation")
    parser.add_argument("--rotate-angles", nargs='+', type=float, default=[90, 180, 270], help="Rotation angles")
    parser.add_argument("--mirror", action="store_true", help="Apply mirroring")
    parser.add_argument("--brightness-contrast", action="store_true", help="Adjust brightness and contrast")
    parser.add_argument("--brightness-range", nargs=2, type=float, default=[0.5, 1.5], help="Brightness adjustment range")
    parser.add_argument("--contrast-range", nargs=2, type=float, default=[0.5, 1.5], help="Contrast adjustment range")
    parser.add_argument("--noise", action="store_true", help="Add noise")
    parser.add_argument("--noise-type", choices=['gaussian', 'salt_and_pepper'], default='gaussian', help="Type of noise to add")
    parser.add_argument("--noise-amount", type=float, default=0.05, help="Amount of noise to add")
    parser.add_argument("--crop", action="store_true", help="Apply random cropping")
    parser.add_argument("--pca", action="store_true", help="Apply PCA color augmentation")
    parser.add_argument("--pca-alpha-std", type=float, default=0.1, help="Standard deviation for PCA color augmentation")
    parser.add_argument("--hsv", action="store_true", help="Convert to HSV and combine")
    parser.add_argument("--lab", action="store_true", help="Convert to LAB and combine")
    parser.add_argument("--fourier", action="store_true", help="Apply Fourier transform")
    parser.add_argument("--gradient", action="store_true", help="Add gradient channels")
    parser.add_argument("--multispectral", action="store_true", help="Convert to multispectral")
    parser.add_argument("--edge", action="store_true", help="Add edge channels")
    parser.add_argument("--green", action="store_true", help="Enhance green channel")
    parser.add_argument("--custom-filter", choices=["emboss", "sharpen"], help="Apply custom filter")

    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        # Update args with config, prioritizing command-line arguments
        for key, value in config.items():
            if not getattr(args, key, None):
                setattr(args, key, value)
    
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(os.path.join(args.output_dir, 'datapalette.log'))
    logger.info(f"Starting DataPalette with pipeline: {args.pipeline}")

    try:
        input_path = process_input(args)
        input_path = apply_tiling(input_path, args)
        execute_pipeline(args, input_path)
        logger.info("DataPalette execution completed successfully")
    except Exception as e:
        logger.exception(f"An error occurred during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()

# python -m datapalette.main --config datapalette_config.yaml

# datapalette-run --output_dir $OUTPUT_DIR --pipeline custom --input-type images --rotate --mirror --fourier --multispectral

# datapalette-run --input $INPUT_DIR --output_dir $OUTPUT_DIR --pipeline gan --input-type video --fps 1

# datapalette-run --output_dir $OUTPUT_DIR --pipeline unet --input-type images