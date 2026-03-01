# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.1] - 2026-03-02

### Fixed
- Fix WSL2 file save issues
- Update README.md and README_zh.md add Special Note for Windows Users, add Environment & Requirements

### Changed
- Add Processing Threads in webui.py and cli.py
- Add Free disk space check in docker-build-push.yml (github action)

## [1.1.0] - 2025-02-23

### Added
- **Multi-Speaker Fine-tuning**: Train a single model with multiple speakers simultaneously. Each speaker gets a unique `spk_id` index and independent speaker embedding.
  - WebUI: Select multiple speakers from the dropdown in Training tab.
  - CLI: Use comma-separated `--speaker_name speaker_a,speaker_b` for `tokenize` and `train` commands.
- **Per-Sample Speaker Encoding**: Speaker embeddings are computed individually per sample to avoid padding artifacts from variable-length reference audio.
- **Automatic JSONL Merge**: Step 3 (Tokenization) automatically merges per-speaker `tts_train.jsonl` files with `speaker_id` field when multiple speakers are selected.

### Changed
- `dataset.py`: Added `speaker_id` field to dataset items and batch dict, `ref_mels` returned as list for variable-length support.
- `sft_12hz.py`: Replaced single `target_speaker_embedding` with per-speaker `speaker_embeddings` dict. Checkpoint saves all speaker embeddings at unique codec indices.
- `data_pipeline.py`: Added optional `speaker_id` to JSONL output.
- `cli.py`: Added `json` and `argparse` imports, multi-speaker support for `tokenize` and `train` subcommands.

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
