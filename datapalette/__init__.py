from __future__ import annotations

__version__ = "0.1.0"

from datapalette.io.batch import load_image, process_directory, save_image
from datapalette.pipelines import (
    DiffusionPipeline,
    GANPipeline,
    SegmentationPipeline,
    TaskPipeline,
)
from datapalette.transforms import (
    BrightnessContrast,
    ConvertColorSpace,
    CustomKernel,
    EdgeChannels,
    Emboss,
    EnhanceGreen,
    FourierTransform,
    GaussianNoise,
    GradientChannels,
    ImageTransform,
    Mirror,
    Multispectral,
    PCAColorAugmentation,
    RandomCrop,
    Resize,
    Rotate,
    SaltPepperNoise,
    Sharpen,
    Tile,
)

__all__ = [
    "__version__",
    # Transforms
    "ImageTransform",
    "Rotate",
    "Mirror",
    "RandomCrop",
    "Tile",
    "Resize",
    "PCAColorAugmentation",
    "ConvertColorSpace",
    "EnhanceGreen",
    "Multispectral",
    "GaussianNoise",
    "SaltPepperNoise",
    "BrightnessContrast",
    "FourierTransform",
    "GradientChannels",
    "EdgeChannels",
    "Emboss",
    "Sharpen",
    "CustomKernel",
    # Pipelines
    "TaskPipeline",
    "GANPipeline",
    "SegmentationPipeline",
    "DiffusionPipeline",
    # I/O
    "process_directory",
    "load_image",
    "save_image",
]
