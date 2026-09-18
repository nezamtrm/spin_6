# Medical Ultrasound Reporting Assistant

AI-powered system for generating structured radiology reports from noisy medical dictations.

## Features

- **Intelligent Noise Filtering**: Handles "نه برگرد عقب" and other filler words
- **Weak Supervision**: Learns from (Audio, Final Report) pairs
- **Content Classification**: Separates relevant content from noise
- **Entity Extraction**: Medical NER from filtered text
- **Template-based Reports**: Structured, consistent outputs

## Quick Start

```bash
# Clone
git clone https://github.com/nezamtrm/BuAli
cd medical-ultrasound-assistant

# Setup
bash scripts/setup_environment.sh

# Install
pip install -r requirements.txt

# Prepare data
# Place your audio and reports in data/raw/
# Then run:
bash scripts/prepare_weak_supervision.sh

# Train models
python training/weak_supervision/step1_create_transcripts.py
python training/weak_supervision/step2_align_to_reports.py
python training/weak_supervision/step3_create_labels.py
python training/weak_supervision/step4_train_content_filter.py

# Run inference
python -m src.pipeline.production --audio exam.wav --output report.json
