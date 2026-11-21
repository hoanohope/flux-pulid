# Configuration Files

This directory contains configuration files used to manage datasets, prompts, and random seeds for the identity-preserving 2D image generation project.

## Files

### 1. selection_log.csv

Records dataset image selection and assignment.

Fields:
- `image_path`: Path to the image.
- `identity_id`: Person identity identifier.
- `subset`: Dataset role (fine-tuning, reference, or evaluation).
- `selection_status`: Selection status.
- `reason`: Explanation for the selection decision.

### 2. prompts.json

Stores prompt templates and random seeds for image generation experiments.

Main fields:
- `version`: Configuration version.
- `pilot_seeds`: Seeds for preliminary experiments.
- `main_seeds`: Seeds for main experiments.
- `prompts`: List of prompt configurations.
- `prompt_id`: Unique prompt identifier.
- `style`: Target 2D illustration style.
- `prompt`: Positive generation prompt.
- `negative_prompt`: Undesired visual characteristics.

## Usage

These configuration files are intended to support consistent and reproducible experiments.

Changes to prompts, seeds, or dataset assignments should be documented.

Actual dataset records and official experiment configurations will be added after the dataset and model setup are finalized.
