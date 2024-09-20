# File: datapalette/main.py

import argparse
import os
import logging
from tqdm import tqdm
from datapalette.pipelines.predefined import gan_pipeline, unet_segmentation_pipeline, diffusion_model_pipeline, custom_pipeline
from datapalette.utils.helpers import setup_logging
from datapalette.preprocessing.advanced import *
from datapalette.augmentation.basic import *
from datapalette.core.functions import extract_frames

logger = logging.getLogger(__name__)

# Set default directories
DEFAULT_INPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

def main():
    parser = argparse.ArgumentParser(description="DataPalette: Image Dataset Creation and Preprocessing")
    parser.add_argument("--input", default=DEFAULT_INPUT_DIR, help="Input directory of images or video file")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for processed images")
    parser.add_argument("--pipeline", choices=['gan', 'unet', 'diffusion', 'custom'], required=True, help="Pipeline to use")
    parser.add_argument("--input-type", choices=['video', 'images'], required=True, help="Type of input (video or images)")
    
    # GAN pipeline specific arguments
    parser.add_argument("--fps", type=int, default=1, help="Frames per second to extract (for video input)")
    parser.add_argument("--crop-size", nargs=2, type=int, default=[256, 256], help="Crop size for GAN pipeline")
    parser.add_argument("--num-crops", type=int, default=5, help="Number of crops per frame (for GAN pipeline)")
    
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
    
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(os.path.join(args.output_dir, 'datapalette.log'))
    logger.info(f"Starting DataPalette with pipeline: {args.pipeline}")

    try:
        if args.input_type == 'video':
            if not os.path.isfile(args.input):
                raise ValueError("For video input, please provide a path to a video file")
            video_frames_dir = os.path.join(args.output_dir, 'video_frames')
            os.makedirs(video_frames_dir, exist_ok=True)
            logger.info(f"Extracting frames from video: {args.input}")
            extract_frames(os.path.dirname(args.input), os.path.basename(args.input), video_frames_dir, fps=args.fps)
            input_for_pipeline = video_frames_dir
        else:
            if not os.path.isdir(args.input):
                raise ValueError("For image input, please provide a path to a directory of images")
            input_for_pipeline = args.input

        if args.pipeline == 'gan':
            if args.input_type != 'video':
                raise ValueError("GAN pipeline requires video input")
            logger.info("Executing GAN pipeline")
            gan_pipeline(input_for_pipeline, args.output_dir, crop_size=tuple(args.crop_size), num_crops=args.num_crops)
        elif args.pipeline == 'unet':
            logger.info("Executing U-Net segmentation pipeline")
            unet_segmentation_pipeline(input_for_pipeline, args.output_dir)
        elif args.pipeline == 'diffusion':
            logger.info("Executing diffusion model pipeline")
            diffusion_model_pipeline(input_for_pipeline, args.output_dir)
        elif args.pipeline == 'custom':
            logger.info("Executing custom pipeline")
            steps = []
            if args.rotate:
                steps.append(('Rotation', lambda x, y: rotate_images(x, y, angles=args.rotate_angles)))
            if args.mirror:
                steps.append(('Mirroring', mirror_images))
            if args.brightness_contrast:
                steps.append(('Brightness/Contrast Adjustment', lambda x, y: adjust_brightness_contrast(x, y, brightness_range=args.brightness_range, contrast_range=args.contrast_range)))
            if args.noise:
                steps.append(('Noise Addition', lambda x, y: add_noise(x, y, noise_type=args.noise_type, amount=args.noise_amount)))
            if args.crop:
                steps.append(('Random Cropping', lambda x, y: random_crop(x, y, crop_size=tuple(args.crop_size))))
            if args.pca:
                steps.append(('PCA Color Augmentation', lambda x, y: pca_color_augmentation(x, y, alpha_std=args.pca_alpha_std)))
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
                steps.append(('Custom Filter', lambda x, y: create_custom_filter_channels(x, y, filter_type=args.custom_filter)))

            for i, (step_name, step_func) in enumerate(tqdm(steps, desc="Executing custom pipeline")):
                logger.info(f"Executing step {i+1}/{len(steps)}: {step_name}")
                step_output_dir = os.path.join(args.output_dir, f'step_{i+1}_{step_name.lower().replace("/", "_")}')
                step_func(input_for_pipeline if i == 0 else os.path.join(args.output_dir, f'step_{i}_{steps[i-1][0].lower().replace("/", "_")}'), step_output_dir)
                input_for_pipeline = step_output_dir

        logger.info("DataPalette execution completed successfully")
    except Exception as e:
        logger.exception(f"An error occurred during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()

# datapalette-run --output_dir $OUTPUT_DIR --pipeline custom --input-type images --rotate --mirror --fourier --multispectral

# datapalette-run --input $INPUT_DIR --output_dir $OUTPUT_DIR --pipeline gan --input-type video --fps 1

# datapalette-run --output_dir $OUTPUT_DIR --pipeline unet --input-type images