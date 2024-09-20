# File: datapalette/preprocessing/advanced.py

import os
import cv2
import numpy as np
import logging
from typing import Optional
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

def pca_color_augmentation(input_dir: str, output_dir: Optional[str] = None, alpha_std: float = 0.1):
    """Apply PCA-based color augmentation and save in /pca_augmentation."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_dir), 'pca_augmentation')
    
    logger.info(f"Applying PCA color augmentation to images in {input_dir}")
    
    def apply_pca_augmentation(img):
        img_reshaped = img.reshape(-1, 3).astype(np.float32)
        
        # Check variance to ensure normalization is stable
        mean = np.mean(img_reshaped, axis=0)
        std = np.std(img_reshaped, axis=0)
        
        if np.any(std == 0):
            logger.warning("Image has low variance, skipping PCA augmentation")
            return img  # Skip augmentation if variance is too low

        # Normalize the image
        img_normalized = (img_reshaped - mean) / std
        
        # Compute the covariance matrix
        cov = np.cov(img_normalized, rowvar=False)
        
        # Regularize the covariance matrix to avoid instability
        cov += np.eye(cov.shape[0]) * 1e-6
        
        try:
            # Compute eigenvalues and eigenvectors
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError as e:
            logger.error(f"Eigenvalue computation failed: {e}, skipping image")
            return img  # Return original image if eigenvalue computation fails
        
        # Sample alphas for augmentation
        alphas = np.random.normal(0, alpha_std, 3)
        delta = np.dot(eigvecs, alphas * eigvals)
        
        # Apply the augmentation and reverse normalization
        img_augmented = img_normalized + delta
        img_augmented = (img_augmented * std) + mean
        img_augmented = np.clip(img_augmented, 0, 255).reshape(img.shape).astype(np.uint8)
        
        return img_augmented

    # Apply PCA augmentation to all images in the input directory
    process_image_batch(input_dir, output_dir, apply_pca_augmentation, desc="Applying PCA augmentation")
    logger.info(f"PCA color augmentation completed. Results saved in {output_dir}")


def convert_to_hsv_and_combine(input_dir: str, output_dir: Optional[str] = None):
    """Convert images to HSV color space, save RGB and HSV separately."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_dir), 'hsv_channels')
    
    os.makedirs(output_dir, exist_ok=True)
    output_rgb_dir = os.path.join(output_dir, 'rgb')
    output_hsv_dir = os.path.join(output_dir, 'hsv')
    os.makedirs(output_rgb_dir, exist_ok=True)
    os.makedirs(output_hsv_dir, exist_ok=True)
    
    logger.info(f"Converting images to HSV and saving RGB and HSV separately in {input_dir}")
    
    def apply_hsv_conversion(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return img, hsv  # Return both RGB and HSV images separately

    def process_and_save(img_file):
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            logger.warning(f"Failed to load image {img_file}. Skipping.")
            return

        # Process the image
        rgb_img, hsv_img = apply_hsv_conversion(img)
        
        # Create output filenames
        base_name = os.path.splitext(img_file)[0]
        output_file_rgb = os.path.join(output_rgb_dir, f"{base_name}.png")
        output_file_hsv = os.path.join(output_hsv_dir, f"{base_name}.png")

        # Save RGB and HSV images
        cv2.imwrite(output_file_rgb, rgb_img)
        cv2.imwrite(output_file_hsv, hsv_img)

    # Iterate through images and process/save them
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_file in tqdm(image_files, desc="Processing and saving images", unit="image"):
        process_and_save(img_file)

    logger.info(f"HSV conversion completed. Results saved in {output_dir}")


def convert_to_lab_and_combine(input_dir: str, output_dir: Optional[str] = None):
    """Convert images to LAB color space and save LAB and RGB separately."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_dir), 'lab_channels')
    
    os.makedirs(output_dir, exist_ok=True)
    output_rgb_dir = os.path.join(output_dir, 'rgb')
    output_lab_dir = os.path.join(output_dir, 'lab')
    os.makedirs(output_rgb_dir, exist_ok=True)
    os.makedirs(output_lab_dir, exist_ok=True)
    
    logger.info(f"Converting images to LAB and saving RGB and LAB separately in {input_dir}")
    
    def apply_lab_conversion(img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        return img, lab  # Return both RGB and LAB images separately

    def process_and_save(img_file):
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            logger.warning(f"Failed to load image {img_file}. Skipping.")
            return

        # Process the image
        rgb_img, lab_img = apply_lab_conversion(img)
        
        # Create output filenames
        base_name = os.path.splitext(img_file)[0]
        output_file_rgb = os.path.join(output_rgb_dir, f"{base_name}.png")
        output_file_lab = os.path.join(output_lab_dir, f"{base_name}.png")

        # Save RGB and LAB images
        cv2.imwrite(output_file_rgb, rgb_img)
        cv2.imwrite(output_file_lab, lab_img)

    # Iterate through images and process/save them
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_file in tqdm(image_files, desc="Converting to LAB and saving separately", unit="image"):
        process_and_save(img_file)

    logger.info(f"LAB conversion completed. Results saved in {output_dir}")


def fourier_transform_channels(input_dir: str, output_dir: Optional[str] = None):
    """Apply Fourier transformation and save channels separately."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_dir), 'fourier_channels')
    
    os.makedirs(output_dir, exist_ok=True)
    output_b_dir = os.path.join(output_dir, 'blue')
    output_g_dir = os.path.join(output_dir, 'green')
    output_r_dir = os.path.join(output_dir, 'red')
    os.makedirs(output_b_dir, exist_ok=True)
    os.makedirs(output_g_dir, exist_ok=True)
    os.makedirs(output_r_dir, exist_ok=True)

    logger.info(f"Applying Fourier transformation to images in {input_dir}")
    
    def apply_fourier_transform(img):
        def process_channel(channel):
            f = np.fft.fft2(channel)
            fshift = np.fft.fftshift(f)
            magnitude = 20 * np.log(np.abs(fshift) + 1)
            magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return magnitude

        b, g, r = cv2.split(img)
        b_fourier = process_channel(b)
        g_fourier = process_channel(g)
        r_fourier = process_channel(r)
        
        return b_fourier, g_fourier, r_fourier  # Return Fourier transformed channels separately

    def process_and_save(img_file):
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)

        if img is None:
            logger.warning(f"Failed to load image {img_file}. Skipping.")
            return

        # Process the image
        b_fourier_img, g_fourier_img, r_fourier_img = apply_fourier_transform(img)

        # Create output filenames
        base_name = os.path.splitext(img_file)[0]
        output_file_b = os.path.join(output_b_dir, f"{base_name}.png")
        output_file_g = os.path.join(output_g_dir, f"{base_name}.png")
        output_file_r = os.path.join(output_r_dir, f"{base_name}.png")

        # Save Fourier transformed images
        cv2.imwrite(output_file_b, b_fourier_img)
        cv2.imwrite(output_file_g, g_fourier_img)
        cv2.imwrite(output_file_r, r_fourier_img)

    # Iterate through images and process/save them
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_file in tqdm(image_files, desc="Applying Fourier transform and saving separately", unit="image"):
        process_and_save(img_file)

    logger.info(f"Fourier transformation completed. Results saved in {output_dir}")


def add_gradient_channels(input_dir: str, output_dir: Optional[str] = None):
    """Add gradient channels using Sobel operators and save magnitude and direction separately."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_dir), 'gradient_channels')
    
    os.makedirs(output_dir, exist_ok=True)
    output_magnitude_dir = os.path.join(output_dir, 'magnitude')
    output_direction_dir = os.path.join(output_dir, 'direction')
    os.makedirs(output_magnitude_dir, exist_ok=True)
    os.makedirs(output_direction_dir, exist_ok=True)
    
    logger.info(f"Adding gradient channels to images in {input_dir}")
    
    def apply_gradient_channels(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        magnitude = cv2.magnitude(grad_x, grad_y)
        direction = cv2.phase(grad_x, grad_y, angleInDegrees=True)
        
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        direction = cv2.normalize(direction, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return magnitude, direction  # Return magnitude and direction separately

    def process_and_save(img_file):
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)

        if img is None:
            logger.warning(f"Failed to load image {img_file}. Skipping.")
            return

        # Process the image
        magnitude_img, direction_img = apply_gradient_channels(img)

        # Create output filenames
        base_name = os.path.splitext(img_file)[0]
        output_file_magnitude = os.path.join(output_magnitude_dir, f"{base_name}.png")
        output_file_direction = os.path.join(output_direction_dir, f"{base_name}.png")

        # Save magnitude and direction images
        cv2.imwrite(output_file_magnitude, magnitude_img)
        cv2.imwrite(output_file_direction, direction_img)

    # Iterate through images and process/save them
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_file in tqdm(image_files, desc="Adding gradient channels and saving separately", unit="image"):
        process_and_save(img_file)

    logger.info(f"Gradient channels added. Results saved in {output_dir}")


def convert_to_multispectral(input_dir: str, output_dir: Optional[str] = None):
    """Simulate multi-spectral imaging and save in /multispectral."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_dir), 'multispectral')
    
    logger.info(f"Converting images to multispectral in {input_dir}")
    
    def apply_multispectral_conversion(img):
        b, g, r = cv2.split(img)
        
        yellow = cv2.addWeighted(g, 0.5, r, 0.5, 0)
        cyan = cv2.addWeighted(b, 0.5, g, 0.5, 0)
        magenta = cv2.addWeighted(b, 0.5, r, 0.5, 0)
        
        nir = cv2.add(r, 50)  # Simulate higher reflectance in NIR
        
        multispectral = np.dstack((b, g, r, yellow, cyan, magenta, nir))
        return multispectral

    process_image_batch(input_dir, output_dir, apply_multispectral_conversion, desc="Converting to multispectral")
    logger.info(f"Multispectral conversion completed. Results saved in {output_dir}")

def add_edge_channels(input_dir: str, output_dir: Optional[str] = None):
    """Perform Canny edge detection and save edges and Laplacian separately."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_dir), 'edge_channels')
    
    os.makedirs(output_dir, exist_ok=True)
    output_edges_dir = os.path.join(output_dir, 'edges')
    output_laplacian_dir = os.path.join(output_dir, 'laplacian')
    os.makedirs(output_edges_dir, exist_ok=True)
    os.makedirs(output_laplacian_dir, exist_ok=True)
    
    logger.info(f"Adding edge channels to images in {input_dir}")
    
    def apply_edge_detection(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(gray, 100, 200)
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))

        return edges, laplacian  # Return edges and Laplacian separately

    def process_and_save(img_file):
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)

        if img is None:
            logger.warning(f"Failed to load image {img_file}. Skipping.")
            return

        # Process the image
        edges_img, laplacian_img = apply_edge_detection(img)

        # Create output filenames
        base_name = os.path.splitext(img_file)[0]
        output_file_edges = os.path.join(output_edges_dir, f"{base_name}.png")
        output_file_laplacian = os.path.join(output_laplacian_dir, f"{base_name}.png")

        # Save edges and Laplacian images
        cv2.imwrite(output_file_edges, edges_img)
        cv2.imwrite(output_file_laplacian, laplacian_img)

    # Iterate through images and process/save them
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_file in tqdm(image_files, desc="Adding edge channels and saving separately", unit="image"):
        process_and_save(img_file)

    logger.info(f"Edge channels added. Results saved in {output_dir}")


def enhance_green_channel(input_dir: str, output_dir: Optional[str] = None):
    """Enhance the green channel using histogram equalization and save in /green_enhanced."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_dir), 'green_enhanced')
    
    logger.info(f"Enhancing green channel of images in {input_dir}")
    
    def apply_green_enhancement(img):
        b, g, r = cv2.split(img)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        g_enhanced = clahe.apply(g)
        
        enhanced = cv2.merge([b, g_enhanced, r])
        return enhanced

    process_image_batch(input_dir, output_dir, apply_green_enhancement, desc="Enhancing green channel")
    logger.info(f"Green channel enhancement completed. Results saved in {output_dir}")

def create_custom_filter_channels(input_dir: str, filter_type: str, output_dir: Optional[str] = None):
    """Apply custom filters and save in /custom_filter_channels."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_dir), 'custom_filter_channels')
    
    logger.info(f"Applying custom filter '{filter_type}' to images in {input_dir}")
    
    def apply_custom_filter(img):
        if filter_type == 'emboss':
            kernel = np.array([[-2, -1, 0],
                               [-1,  1, 1],
                               [ 0,  1, 2]])
        elif filter_type == 'sharpen':
            kernel = np.array([[ 0, -1,  0],
                               [-1,  5, -1],
                               [ 0, -1,  0]])
        else:
            logger.error(f"Unsupported filter type: {filter_type}")
            raise ValueError("Unsupported filter type. Choose 'emboss' or 'sharpen'.")
        
        filtered = cv2.filter2D(img, -1, kernel)
        combined = np.dstack((img, filtered))
        return combined

    process_image_batch(input_dir, output_dir, apply_custom_filter, desc=f"Applying {filter_type} filter")
    logger.info(f"Custom filter '{filter_type}' applied. Results saved in {output_dir}")