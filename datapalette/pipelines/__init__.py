# File: datapalette/pipelines/__init__.py

from .predefined import gan_pipeline, unet_segmentation_pipeline, diffusion_model_pipeline, custom_pipeline

__all__ = ['gan_pipeline', 'unet_segmentation_pipeline', 'diffusion_model_pipeline', 'custom_pipeline']