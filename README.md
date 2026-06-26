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

## :pencil: Supported Methods
FDeID-Toolbox currently supports the following face de-identification algorithms:
| Category | Method | Config Key | Paper Link | 
|----------|--------|------------| ------------| 
| Naive | Gaussian Blur | `blur` |  ---  |
| Naive | Pixelation | `pixelate` |  ---  |
| Naive | Black Mask | `mask` |  ---  |
| K-Same | k-Same-Average | `average` | <small>[Preserving Privacy by De-identifying Face Images](https://ieeexplore.ieee.org/document/1377174)<small> |
| K-Same | k-Same-Select | `select` | <small>[Integrating Utility into Face De-identification](https://link.springer.com/chapter/10.1007/11767831_15)<small> |
| K-Same | k-Same-Furthest | `furthest` | <small>[Face De-identification with Perfect Privacy Protection](https://ieeexplore.ieee.org/document/6859756)<small> | 
| Generative | CIAGAN | `ciagan` | <small>[CIAGAN: Conditional Identity Anonymization Generative Adversarial Networks](https://openaccess.thecvf.com/content_CVPR_2020/html/Maximov_CIAGAN_Conditional_Identity_Anonymization_Generative_Adversarial_Networks_CVPR_2020_paper.html)<small> |
| Generative | AMT-GAN | `amtgan` | <small>[Protecting Facial Privacy: Generating Adversarial Identity Masks via Style-Robust Makeup Transfer](https://openaccess.thecvf.com/content/CVPR2022/html/Hu_Protecting_Facial_Privacy_Generating_Adversarial_Identity_Masks_via_Style-Robust_Makeup_CVPR_2022_paper.html)<small> |
| Generative | Adv-Makeup | `advmakeup` | <small>[Adv-Makeup: A New Imperceptible and Transferable Attack on Face Recognition](https://arxiv.org/abs/2105.03162)<small> |
| Generative | WeakenDiff | `weakendiff` | <small>[Enhancing Facial Privacy Protection via Weakening Diffusion Purification](https://openaccess.thecvf.com/content/CVPR2025/html/Salar_Enhancing_Facial_Privacy_Protection_via_Weakening_Diffusion_Purification_CVPR_2025_paper.html)<small> |
| Generative | DeID-rPPG | `deid_rppg` | <small>[De-identification of Facial Videos while Preserving Remote Physiological Utility](https://papers.bmvc2023.org/0230.pdf)<small> |
| Generative | G2Face | `g2face` | <small>[G²Face: High-Fidelity Reversible Face Anonymization via Generative and Geometric Priors](https://ieeexplore.ieee.org/abstract/document/10644096?casa_token=m_NCPo_OrA4AAAAA:Dg8FslVjBG_UtThsgdXdcSIwnbxOA4S1i5NqNvQRZwDCqhL58BmIeey78288H29kbzcmf6pnfE5i)<small> |
| Adversarial | PGD | `pgd` | <small>[Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)<small> |
| Adversarial | MI-FGSM | `mifgsm` | <small>[Boosting Adversarial Attacks With Momentum](https://openaccess.thecvf.com/content_cvpr_2018/html/Dong_Boosting_Adversarial_Attacks_CVPR_2018_paper.html)<small> |
| Adversarial | TI-DIM | `tidim` | <small>[Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks](https://openaccess.thecvf.com/content_CVPR_2019/html/Dong_Evading_Defenses_to_Transferable_Adversarial_Examples_by_Translation-Invariant_Attacks_CVPR_2019_paper.html)<small> |
| Adversarial | TI-PIM | `tipim` | <small>[Towards Face Encryption by Generating Adversarial Identity Masks](https://openaccess.thecvf.com/content/ICCV2021/html/Yang_Towards_Face_Encryption_by_Generating_Adversarial_Identity_Masks_ICCV_2021_paper.html)<small> |
| Adversarial | Chameleon | `chameleon` | <small>[Personalized Privacy Protection Mask Against Unauthorized Facial Recognition](https://link.springer.com/chapter/10.1007/978-3-031-73007-8_25)<small> |


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