<img src="./assets/logo.png" width=100%>

# Face De-identification Toolbox

A modular face de-identification toolbox for privacy-preserving facial analysis research.

## :bell: Bugs, Fixes, and New Features
We welcome community feedback and contributions to help improve this toolbox.  
* [x] Pretrained model uploading and download link providing.
* [x] Generative methods uploading.
* [x] Adversarial methods uploading.
* [x] Naive methods uploading.
* [x] $k$-Same family methods uploading.


## :rocket: Quick Start 

### Environment Setup
**FDeID-Toolbox** is built exclusively on PyTorch, ensuring a lightweight and unified environment. No complex C++ extensions or conflicting frameworks.

Note that the download link of all pretrained models: [link](https://huggingface.co/buckets/huiwei25/FDeID-Toolbox). 
```bash
# 1. Clone the repo
git clone https://github.com/infraface/FDeID-Toolbox.git
cd FDeID-Toolbox

# 2. Install dependencies (Only PyTorch and standard vision libs)
pip install -r requirements.txt

# 3. Download the pretrained models and place them in the weight/ folder
unzip weight.zip
```

### Running an Experiment

Every experiment is fully specified by a single YAML config file. CLI arguments can override any config value.

```bash
# Run from config only
python scripts/run_naive_deid.py --config configs/naive/blur_lfw.yaml

# Override specific parameters
python scripts/run_naive_deid.py --config configs/naive/blur_lfw.yaml --kernel_size 100 --max_images 50

# Run without config (traditional CLI usage still works)
python scripts/run_naive_deid.py --dataset lfw --method blur --save_dir runs/inference/blur_lfw
```

### Running Evaluation

```bash
python scripts/eval_privacy_lfw.py --config configs/eval/privacy_lfw.yaml --deid_dir runs/inference/blur_lfw
python scripts/eval_quality.py --config configs/eval/quality.yaml --original_dir /path/to/lfw --deid_dir runs/inference/blur_lfw
```

## :pencil: Available Methods

| Category | Method | Config Key |
|----------|--------|------------|
| Naive | Gaussian Blur | `blur` |
| Naive | Pixelation | `pixelate` |
| Naive | Black Mask | `mask` |
| Generative | CIAGAN | `ciagan` |
| Generative | AMT-GAN | `amtgan` |
| Generative | Adv-Makeup | `advmakeup` |
| Generative | WeakenDiff | `weakendiff` |
| Generative | DeID-rPPG | `deid_rppg` |
| Generative | G2Face | `g2face` |
| Adversarial | PGD | `pgd` |
| Adversarial | MI-FGSM | `mifgsm` |
| Adversarial | TI-DIM | `tidim` |
| Adversarial | TI-PIM | `tipim` |
| Adversarial | Chameleon | `chameleon` |
| K-Same | k-Same-Average | `average` |
| K-Same | k-Same-Select | `select` |
| K-Same | k-Same-Furthest | `furthest` |

## :triangular_ruler: Evaluation Metrics 

| Category | Metric | Script |
|----------|--------|--------|
| Privacy | Verification accuracy, TAR@FAR, PSR | `eval_privacy_lfw.py`, `eval_privacy_agedb.py` |
| Quality | PSNR, SSIM, LPIPS, FID, NIQE | `eval_quality.py` |
| Utility - Age | MAE | `eval_age.py` |
| Utility - Gender | Accuracy | `eval_gender.py` |
| Utility - Expression | Accuracy | `eval_expression.py` |
| Utility - Landmark | NME | `eval_landmark.py` |
| Utility - Ethnicity | Accuracy | `eval_ethnicity.py` |
| Utility - rPPG | Heart rate MAE, RMSE | `eval_rppg_utility.py` |

## :page_facing_up: Supported Datasets

| Dataset | Evaluation Metric |
|---------|------|
| LFW | `Privacy`, `Quality` |
| AgeDB | `Privacy`, `Utility-Age`, `Quality` |
| AffectNet | `Utility-Expression` |
| CelebA-HQ | `Utility-Landmark` |
| FairFace | `Utility-Gender`, `Utility-Ethnicity` |
| PURE | `Utility-rPPG` |

## :hearts: Acknowledgments
We thank the authors and maintainers of open-source repositories and pretrained models that our toolbox builds on or reimplements. We acknowledge work related to:

- RetinaFace and Dlib for face detection and landmark localization
- ArcFace, CosFace, and AdaFace for face recognition
- FairFace, POSTER, HRNet, and FactorizePhys for utility evaluation
- k-Same-Average, k-Same-Select, and k-Same-Furthest
- PGD, MI-FGSM, TI-DIM, TIP-IM, and Chameleon
- CIAGAN, AMT-GAN, Adv-Makeup, WeakenDiff, DeID-rPPG, and G$^{2}$Face

Please support these original projects by citing their papers and visiting their repositories.

## :balloon: Citation
If you find our work useful, please kindly cite as:
```
@article{wei2026fdeid,
  title={FDeID-Toolbox: Face De-Identification Toolbox},
  author={Wei, Hui and Yu, Hao and Zhao, Guoying},
  journal={arXiv preprint arXiv:2603.13121},
  year={2026}
}
```