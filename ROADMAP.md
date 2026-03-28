# DataPalette Roadmap

## Current State (v0.1.0)

- Clean sklearn-compatible architecture: 18 transform classes composable with `sklearn.pipeline.Pipeline`
- Task-aware pipelines returning `(X, y)`: GANPipeline, SegmentationPipeline, DiffusionPipeline
- Built-in dataset support (MNIST / Fashion-MNIST via torchvision)
- Segmentation mask generators (binary, multiclass, instance)
- PyTorch Dataset adapter with lazy loading
- Modern packaging (pyproject.toml, py.typed, CLI entry point)
- 61 tests passing, ruff clean, mypy clean

### Known Gaps

- Test coverage is shallow (~60% estimated) — mostly "doesn't crash" and shape checks
- CLI only tested for `--help` and import, no end-to-end test with real images
- Built-in dataset paths (MNIST, segmentation masks) untested in CI (torch skipped)
- sklearn `FutureWarning`: calling `.transform()` without `.fit()` on stateless transforms will break in sklearn 1.8
- `adapters/torch.py` only tested when torch is installed

---

## v0.1.1 — Harden What Exists

- Fix the sklearn FutureWarning: call `pipe.fit([dummy_image])` in pipelines, or override `__sklearn_tags__` to mark transforms as stateless
- Add `pytest --cov` to CI and target 85%+ line coverage
- Add edge-case tests: grayscale input, single-channel, RGBA, empty directory, corrupt image files
- Add integration test using the sample images in `data/`

## v0.1.2 — Developer Experience

- `datapalette list-transforms` CLI command (prints available transforms with params)
- Config-driven pipelines: `datapalette --config examples/config.yaml` working end-to-end with new transform classes
- Verbose / dry-run mode for CLI
- Progress callbacks for programmatic use

## v0.2.0 — Expand Transform Library

- **Geometric**: ElasticDeformation, Perspective, Affine
- **Color**: ColorJitter (unified hue/sat/brightness/contrast), ChannelShuffle, CLAHE (generic, not just green)
- **Augmentation policies**: RandAugment, AutoAugment-style random composition
- **Deterministic mode**: seeded RNG per transform for reproducibility (critical for segmentation mask pairing)

## v0.3.0 — Real Dataset Support

- Support common segmentation datasets (VOC, COCO format) — not just MNIST
- Image-mask paired loading with automatic format detection
- Dataset splitting utilities (train/val/test with stratification)
- HuggingFace Datasets integration as an alternative to torchvision

## v0.4.0 — Performance

- Batch transforms using vectorized numpy (currently loops over images one-by-one)
- Optional GPU acceleration via cupy or kornia for heavy transforms
- Multiprocessing in `process_directory`
- Lazy pipeline evaluation (don't materialize intermediate results)

## v1.0.0 — Stable Release

- Stable public API with deprecation policy
- Comprehensive docs (Sphinx or MkDocs with API reference)
- Benchmarks against albumentations / torchvision transforms
- Published to PyPI

---

## Strategic Direction

The transform library space is crowded (albumentations, torchvision, imgaug). DataPalette's differentiator is the **task-aware pipeline concept** — `GANPipeline(dataset='mnist')` returns `(X, y)` ready for training.

Rather than competing on individual transform quality, the focus is **zero-config dataset preparation for common deep learning tasks**:

1. More built-in datasets and tasks (object detection, super-resolution, style transfer)
2. One-liner experiment setup: `X, y = GANPipeline(dataset='celeba', size=128).load_and_transform()`
3. Integration with training loops (PyTorch Lightning, HuggingFace Trainer)
