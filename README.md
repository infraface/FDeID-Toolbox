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
<table>
  <thead>
    <tr>
      <th><sub>Category</sub></th>
      <th><sub>Method</sub></th>
      <th><sub>Config Key</sub></th>
      <th><sub>Paper Link</sub></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><sub>Naive</sub></td>
      <td><sub>Gaussian Blur</sub></td>
      <td><sub><code>blur</code></sub></td>
      <td><sub>---</sub></td>
    </tr>
    <tr>
      <td><sub>Naive</sub></td>
      <td><sub>Pixelation</sub></td>
      <td><sub><code>pixelate</code></sub></td>
      <td><sub>---</sub></td>
    </tr>
    <tr>
      <td><sub>Naive</sub></td>
      <td><sub>Black Mask</sub></td>
      <td><sub><code>mask</code></sub></td>
      <td><sub>---</sub></td>
    </tr>
    <tr>
      <td><sub>K-Same</sub></td>
      <td><sub>k-Same-Average</sub></td>
      <td><sub><code>average</code></sub></td>
      <td><sub><a href="https://ieeexplore.ieee.org/document/1377174">Preserving Privacy by De-identifying Face Images</a></sub></td>
    </tr>
    <tr>
      <td><sub>K-Same</sub></td>
      <td><sub>k-Same-Select</sub></td>
      <td><sub><code>select</code></sub></td>
      <td><sub><a href="https://link.springer.com/chapter/10.1007/11767831_15">Integrating Utility into Face De-identification</a></sub></td>
    </tr>
    <tr>
      <td><sub>K-Same</sub></td>
      <td><sub>k-Same-Furthest</sub></td>
      <td><sub><code>furthest</code></sub></td>
      <td><sub><a href="https://ieeexplore.ieee.org/document/6859756">Face De-identification with Perfect Privacy Protection</a></sub></td>
    </tr>
    <tr>
      <td><sub>Generative</sub></td>
      <td><sub>CIAGAN</sub></td>
      <td><sub><code>ciagan</code></sub></td>
      <td><sub><a href="https://openaccess.thecvf.com/content_CVPR_2020/html/Maximov_CIAGAN_Conditional_Identity_Anonymization_Generative_Adversarial_Networks_CVPR_2020_paper.html">CIAGAN: Conditional Identity Anonymization Generative Adversarial Networks</a></sub></td>
    </tr>
    <tr>
      <td><sub>Generative</sub></td>
      <td><sub>AMT-GAN</sub></td>
      <td><sub><code>amtgan</code></sub></td>
      <td><sub><a href="https://openaccess.thecvf.com/content/CVPR2022/html/Hu_Protecting_Facial_Privacy_Generating_Adversarial_Identity_Masks_via_Style-Robust_Makeup_CVPR_2022_paper.html">Protecting Facial Privacy: Generating Adversarial Identity Masks via Style-Robust Makeup Transfer</a></sub></td>
    </tr>
    <tr>
      <td><sub>Generative</sub></td>
      <td><sub>Adv-Makeup</sub></td>
      <td><sub><code>advmakeup</code></sub></td>
      <td><sub><a href="https://arxiv.org/abs/2105.03162">Adv-Makeup: A New Imperceptible and Transferable Attack on Face Recognition</a></sub></td>
    </tr>
    <tr>
      <td><sub>Generative</sub></td>
      <td><sub>WeakenDiff</sub></td>
      <td><sub><code>weakendiff</code></sub></td>
      <td><sub><a href="https://openaccess.thecvf.com/content/CVPR2025/html/Salar_Enhancing_Facial_Privacy_Protection_via_Weakening_Diffusion_Purification_CVPR_2025_paper.html">Enhancing Facial Privacy Protection via Weakening Diffusion Purification</a></sub></td>
    </tr>
    <tr>
      <td><sub>Generative</sub></td>
      <td><sub>DeID-rPPG</sub></td>
      <td><sub><code>deid_rppg</code></sub></td>
      <td><sub><a href="https://papers.bmvc2023.org/0230.pdf">De-identification of Facial Videos while Preserving Remote Physiological Utility</a></sub></td>
    </tr>
    <tr>
      <td><sub>Generative</sub></td>
      <td><sub>G2Face</sub></td>
      <td><sub><code>g2face</code></sub></td>
      <td><sub><a href="https://ieeexplore.ieee.org/abstract/document/10644096?casa_token=m_NCPo_OrA4AAAAA:Dg8FslVjBG_UtThsgdXdcSIwnbxOA4S1i5NqNvQRZwDCqhL58BmIeey78288H29kbzcmf6pnfE5i">G²Face: High-Fidelity Reversible Face Anonymization via Generative and Geometric Priors</a></sub></td>
    </tr>
    <tr>
      <td><sub>Adversarial</sub></td>
      <td><sub>PGD</sub></td>
      <td><sub><code>pgd</code></sub></td>
      <td><sub><a href="https://arxiv.org/abs/1706.06083">Towards Deep Learning Models Resistant to Adversarial Attacks</a></sub></td>
    </tr>
    <tr>
      <td><sub>Adversarial</sub></td>
      <td><sub>MI-FGSM</sub></td>
      <td><sub><code>mifgsm</code></sub></td>
      <td><sub><a href="https://openaccess.thecvf.com/content_cvpr_2018/html/Dong_Boosting_Adversarial_Attacks_CVPR_2018_paper.html">Boosting Adversarial Attacks With Momentum</a></sub></td>
    </tr>
    <tr>
      <td><sub>Adversarial</sub></td>
      <td><sub>TI-DIM</sub></td>
      <td><sub><code>tidim</code></sub></td>
      <td><sub><a href="https://openaccess.thecvf.com/content_CVPR_2019/html/Dong_Evading_Defenses_to_Transferable_Adversarial_Examples_by_Translation-Invariant_Attacks_CVPR_2019_paper.html">Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks</a></sub></td>
    </tr>
    <tr>
      <td><sub>Adversarial</sub></td>
      <td><sub>TI-PIM</sub></td>
      <td><sub><code>tipim</code></sub></td>
      <td><sub><a href="https://openaccess.thecvf.com/content/ICCV2021/html/Yang_Towards_Face_Encryption_by_Generating_Adversarial_Identity_Masks_ICCV_2021_paper.html">Towards Face Encryption by Generating Adversarial Identity Masks</a></sub></td>
    </tr>
    <tr>
      <td><sub>Adversarial</sub></td>
      <td><sub>Chameleon</sub></td>
      <td><sub><code>chameleon</code></sub></td>
      <td><sub><a href="https://link.springer.com/chapter/10.1007/978-3-031-73007-8_25">Personalized Privacy Protection Mask Against Unauthorized Facial Recognition</a></sub></td>
    </tr>
  </tbody>
</table>


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
