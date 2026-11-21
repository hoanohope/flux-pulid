# Experimental Protocol

## 1. Purpose and Scope

### 1.1. Purpose

This document defines the experimental protocol for the AI image generation system developed in this project.

The purpose of this protocol is to establish consistent and reproducible procedures for data preparation, image generation, model configuration, checkpoint selection, and evaluation.

The protocol aims to ensure that experimental results are reliable, comparable, and reproducible across different experiments.

### 1.2. Scope

This protocol covers the following experimental components:

- Dataset preparation and identity-based data splitting.
- Prompt and random seed configuration.
- Model checkpoint selection.
- Identity preservation evaluation.
- Generated image quality evaluation.
- Experiment logging and result reporting.

### 1.3. General Principles

All experiments must follow these principles:

1. **Reproducibility:** Experimental configurations, prompts, seeds, and model checkpoints must be recorded.
2. **Consistency:** The same evaluation conditions must be applied when comparing different configurations.
3. **Data separation:** Data used for training, validation, and testing must follow the predefined splitting rules.
4. **Transparency:** All data selection decisions, experimental settings, and evaluation results must be documented.
5. **Fair comparison:** Model comparisons must use consistent input conditions and evaluation procedures whenever applicable.
## 2. Dataset and Data Usage

### 2.1. Candidate Datasets

The following datasets are considered potential sources of facial images for experimental purposes:

| Dataset | Intended Purpose | Status |
|---|---|---|
| FFHQ | High-quality facial image samples | License verification required |
| CelebA-HQ | Facial image samples for model evaluation | License verification required |
| FairFace | Face attribute diversity analysis | License verification required |

The final dataset selection will depend on data availability, compatibility with the project requirements, and license conditions.

### 2.2. Data Usage Requirements

Before using any dataset, the project team must:

1. Identify the original dataset source.
2. Review the dataset license and terms of use.
3. Verify whether academic and research use is permitted.
4. Check whether redistribution or sharing of the dataset is restricted.
5. Document the source and license information.

No dataset will be considered approved for use until its license conditions have been reviewed.

### 2.3. Data Integrity and Privacy

- Original images must be preserved without unauthorized modification.
- Each image must be associated with a valid identity label when identity-based experiments are performed.
- Personal information unrelated to the experiment must not be collected.
- Dataset access and usage must comply with the applicable license and privacy requirements.

### 2.4. Dataset Availability

If the required datasets are not available during the initial implementation phase, the team may develop and test the data-processing pipeline using synthetic directory structures or mock metadata.

Mock data must not be treated as real experimental data, and no experimental conclusions may be drawn from it.
## 3. Identity-based Data Splitting

The dataset must be organized to support identity-preserving 2D image generation.

### 3.1. Reference Images

Reference images are real-person face images provided as input to the generation pipeline. They are used to guide the model in preserving the identity of the input person while generating 2D-style images.

### 3.2. Fine-tuning Data

Images used for fine-tuning must be documented separately from reference and evaluation images. The identity and source of each image must be recorded when this information is available.

### 3.3. Evaluation Data

Evaluation images are used to assess whether generated 2D images preserve the identity of the reference person.

When possible, evaluation images should be different from the images used for fine-tuning to reduce data leakage and provide a more reliable evaluation.

### 3.4. Data Separation and Tracking

The project must record the role of each image, including:

- Fine-tuning image
- Reference image
- Evaluation image

The `selection_log.csv` file should contain the following fields:

- `image_path`
- `identity_id`
- `subset`
- `selection_status`
- `reason`

Images and identities must be tracked carefully to identify possible overlap between fine-tuning and evaluation data.

No actual dataset split should be finalized until the dataset and fine-tuning strategy have been confirmed.

### 3.5. Split Validation

Before running official experiments, verify that:

- All image paths exist.
- Image roles are correctly assigned.
- Identity labels are valid when available.
- Fine-tuning and evaluation data are appropriately separated.
- Any overlap or limitation is documented.
## 4. Prompt and Seed Configuration

### 4.1. Objective

This section defines the prompt and random seed configuration used in image generation experiments.

The purpose is to ensure consistent generation conditions and reproducible experimental results.

### 4.2. Prompt Configuration

Prompts must be defined and documented before the main evaluation.

The prompt configuration should include:

- Prompt ID: A unique identifier for each prompt.
- Prompt text: The complete text input provided to the generation model.
- Image style: The intended visual style.
- Generation purpose: Pilot testing or main evaluation.

Prompts should remain unchanged during a controlled comparison unless prompt variation is the specific experimental factor being studied.

### 4.3. Pilot and Main Prompts

Two prompt categories will be maintained:

- Pilot prompts: Used during preliminary testing to verify the generation pipeline and identify configuration issues.
- Main prompts: Used for the official evaluation after the pilot phase is completed.

The final main prompts must be documented and frozen before the official evaluation begins.

### 4.4. Random Seed Configuration

A fixed random seed must be used whenever reproducible generation is required.

The seed configuration must distinguish between:

- Pilot seed: Used for preliminary testing and debugging.
- Main seed: Used for official experiments and controlled comparisons.

The selected seed values must be recorded in the experiment configuration.

### 4.5. Generation Configuration

The following generation parameters must also be recorded:

| Parameter | Description |
|---|---|
| model_name | Name of the generation model |
| checkpoint | Model checkpoint or version |
| seed | Random seed |
| prompt_id | Identifier of the prompt |
| width | Output image width |
| height | Output image height |
| steps | Number of inference steps |
| guidance_scale | Guidance scale |
| generation_date | Date of generation |

### 4.6. Configuration Management

All approved prompts and seed configurations must be stored in `configs/prompts.json`.

Any changes to the main prompts, seeds, or generation parameters must be documented with a reason and configuration version.

The official evaluation must use the frozen configuration to ensure consistency.
## 5. Checkpoint Selection

### 5.1. Objective

This section defines the procedure for selecting model checkpoints for image generation experiments.

The selection process aims to ensure that the chosen checkpoint provides reliable generation quality and identity preservation performance.

### 5.2. Candidate Checkpoints

All candidate checkpoints must be documented before evaluation.

The following information must be recorded:

- Checkpoint name and version.
- Base model name.
- Model configuration.
- Source or download location.
- Any additional components, such as LoRA weights or identity-conditioning modules.

### 5.3. Selection Criteria

Candidate checkpoints should be assessed based on the following criteria:

1. **Image Quality:** Generated images should have clear details, appropriate composition, and minimal visual artifacts.
2. **Identity Preservation:** The generated face should maintain the identity characteristics of the reference image.
3. **Prompt Alignment:** The generated image should follow the intended prompt.
4. **Generation Stability:** The checkpoint should produce consistent results under controlled generation conditions.
5. **Resource Requirements:** Memory usage and generation time should be considered.

### 5.4. Selection Procedure

1. Identify the available candidate checkpoints.
2. Apply the same pilot prompts and generation conditions to each candidate whenever applicable.
3. Generate sample images for preliminary inspection.
4. Evaluate the outputs using the predefined selection criteria.
5. Document the comparison results.
6. Select the checkpoint that best satisfies the project requirements.

### 5.5. Checkpoint Selection Rules

- Checkpoint selection must be based on documented evaluation results rather than a single visually appealing output.
- The selected checkpoint must be compatible with the project pipeline.
- The selected checkpoint and its configuration must remain fixed during a controlled experiment.
- Any checkpoint replacement must include a documented reason and a new evaluation record.

### 5.6. Selection Record

The selected checkpoint information must be recorded, including:

- Selected checkpoint name.
- Model version.
- Selection date.
- Evaluation configuration.
- Selection rationale.

If no candidate checkpoint has been evaluated, the selection status must remain pending.
## 6. Evaluation Metrics

The evaluation focuses on four main aspects of identity-preserving 2D image generation.

### 6.1. Identity Preservation

Identity preservation measures how well the generated 2D image retains the identity of the person in the reference image.

Face embeddings may be extracted using a face recognition model such as InsightFace. Cosine similarity can then be calculated between the reference face embedding and the generated face embedding.

Higher similarity indicates stronger identity consistency. However, embedding-based evaluation should be interpreted carefully because converting real faces into 2D illustrations may reduce face recognition accuracy.

If face detection or embedding extraction fails, the result must be recorded rather than silently excluded.

### 6.2. 2D Style Consistency

This metric evaluates whether the generated image follows the intended 2D illustration style, such as anime or cartoon.

Evaluation may include visual inspection and comparison with the requested style.

### 6.3. Prompt Alignment

Prompt alignment measures how well the generated image follows the user's text description.

CLIP-based similarity may be used as an optional quantitative metric, together with human evaluation.

### 6.4. Image Quality

Image quality evaluates visual clarity, composition, artifacts, and overall aesthetic quality.

Human evaluation may be used to assess the generated images.

### 6.5. Human Evaluation

Human evaluators may rate generated images using a 1–5 scale:

- 1: Very poor
- 2: Poor
- 3: Acceptable
- 4: Good
- 5: Excellent

Each evaluation aspect should be rated separately.

### 6.6. Reporting Results

All evaluation results must include the model configuration, reference image information, prompt, seed, and evaluation method.

The project must report failed face detection cases and other evaluation limitations.

No evaluation scores should be fabricated or reported before actual experiments are conducted.
## 7. Experiment Reproducibility and Logging

### 7.1. Objective

This section defines the requirements for recording experimental configurations and results.

The purpose is to ensure that experiments can be tracked, compared, and reproduced whenever possible.

### 7.2. Experiment Configuration

Each experiment must record the following information:

- Experiment ID.
- Experiment date and time.
- Model name and checkpoint version.
- Dataset and identity information.
- Prompt ID and prompt text.
- Random seed.
- Image width and height.
- Number of inference steps.
- Guidance scale.
- Hardware and software environment.

### 7.3. Output Management

Generated images and related experiment outputs must be stored in an organized directory structure.

Each experiment should have a unique identifier to avoid accidental overwriting.

The output directory should contain:

- Generated images.
- Experiment configuration.
- Evaluation results.
- Relevant logs and error messages.

### 7.4. Logging Requirements

The team must maintain experiment logs containing:

- Experiment ID.
- Execution status.
- Model checkpoint.
- Prompt ID.
- Random seed.
- Generation parameters.
- Generation time.
- Output image paths.
- Evaluation metrics, when available.
- Errors or issues encountered.

### 7.5. Reproducibility Requirements

To reproduce an experiment, the team should preserve:

1. The exact prompt and generation parameters.
2. The random seed.
3. The model checkpoint and its version.
4. The relevant dataset version and data split.
5. The software dependencies and execution environment.

A fixed seed alone does not guarantee identical outputs across different hardware, software versions, or model implementations.

### 7.6. Result Integrity

- Original experiment logs must be preserved.
- Failed experiments must be recorded when relevant.
- Missing results must not be replaced with fabricated values.
- Changes to configurations must be documented.
- Experimental results must be traceable to their corresponding configurations and outputs.
## 8. Protocol Versioning and Updates

This protocol may be updated when the project requirements, datasets, models, or evaluation methods change.

Each update must include:

- Version number
- Update date
- Description of changes
- Reason for the update

Changes to datasets, model configurations, prompts, or evaluation metrics must be documented to maintain experiment reproducibility.

Previous experiment records must be preserved. Results obtained under different protocol versions should not be directly compared without explaining the differences.

The current protocol version should be recorded in each official experiment log.
