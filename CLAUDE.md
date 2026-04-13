# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal
Develop a modular face de-identification toolbox for privacy-preserving face recognition research on the LUMI supercomputer platform.


## Architecture Overview
The toolbox is designed with modularity, extensibility, and reproducibility as core principles. It comprises four interconnected modules:

1. **Data Module** (`core/data/`): Unified data loaders and preprocessing pipelines for diverse face datasets (LFW, AgeDB, etc.) with standardized annotation formats
2. **Method Module** (`core/fdeid/`): FDeID algorithms spanning naive techniques, generative models, and adversarial perturbation methods, all adhering to `BaseDeIdentifier` interface
3. **Pipeline Module**: Pre-Processing (Face detection, Alignment), face de-identification methods, and Post-Processing (Reinsertion, Blending).
4. **Evaluation Module** (`core/eval/`, `core/identity/`, `core/utility/`): Privacy protection metrics (ArcFace, CosFace, AdaFace), utility preservation measures (age, gender, Expression, Landmark, ethnicity, rPPG), and visual quality assessments (PSNR, SSIM, FID, NIQE, LPIPS).

## Output Structure
- Training: `runs/train/exp_YYYYMMDD_HHMMSS_JOBID/`
- Evaluation: `runs/eval/exp_YYYYMMDD_HHMMSS_JOBID/`
- Inference: `runs/inference/exp_YYYYMMDD_HHMMSS_JOBID/`
- Logs: `logs/jobname_JOBID.out` and `logs/jobname_JOBID.err`

Appended SLURM_JOB_ID (or PID for non-SLURM runs) to the directory name for avoiding collisions when multiple SLURM jobs started at the same second.

## Important Details
The /flash/project_462001188/project/toolbox/toolbox_anonymous project is based on /flash/project_462001188/project/toolbox/toolbox_v1.2. Due to I plan to release the codebase on Github, I create this project.