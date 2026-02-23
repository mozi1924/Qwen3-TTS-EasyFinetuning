# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-02-23

### Added
- **Unified CLI**: Introduced `src/cli.py` as a single entry point for data preparation, training, and inference.
- **Modern WebUI**: Redesigned the Gradio interface for better aesthetics and simplified workflow.
- **Docker Support**: Added `Dockerfile` and `docker-compose.yml` for easy GHCR deployment and local building.
- **Tensorboard Integration**: Real-time training visualization with Mel spectrogram support.
- **Checkpoint Management**: Automatic checkpoint resumption and optimized storage (handling `accelerate_state`).
- **Path Resolution**: Robust path handling for both Docker and local environments.
- **Experimental Features**: Added multi-core CPU processing options and support for different model sizes (0.6B/1.7B).

### Changed
- Refactored pipeline modules into internal Python generators for seamless WebUI integration.
- Optimized VRAM management using multiprocessing to ensure clean GPU memory release between steps.
- Improved logging system to categorize logs by experiment name.

### Fixed
- Fixed Tensorboard initialization errors in Docker environments.
- Resolved pathing issues with intermediate JSONL files during data preparation.
- Addressed progress bar flickering and UI sync issues during training.
