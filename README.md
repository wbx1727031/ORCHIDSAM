# OrchidSAM

<!-- TODO: 获得 DOI 后取消注释并补全
[![DOI](https://zenodo.org/badge/DOI/your-doi-here.svg)](https://doi.org/your-doi-here)
-->
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Pretrained weights and inference scripts for prompt-free floral organ semantic segmentation of Chinese *Cymbidium* orchid images.

This repository accompanies the paper:

> **Floral Organ Semantic Segmentation from Chinese Cymbidium Orchid Images: Adapted Segment Anything Model Fine-tuning versus Task-Specific Model Construction**

## Overview

Chinese *Cymbidium* orchids show rich diversity in floral shape and color. Extracting organ-level phenotypic traits from images still relies largely on manual evaluation. This project provides tools to automate that step by segmenting four floral organ types in orchid images: **sepal**, **petal**, **labellum**, and **gynostemium**.

The repository includes:

- **Adapted SAM (OrchidSAM)**: The Segment Anything model (SAM) adapted for prompt-free semantic segmentation. Prompt-driven tokens are replaced with organ-type-specific queries, so no manual point or box prompts are needed. We fine-tune the adapted model with an optimized loss function.
- **Conventional deep learning baselines**: CCNet, DMNet, UNViT, and FCN, implemented with the PaddleSeg framework.

## Repository contents

| File | Description |
| --- | --- |
| `SAM model predict.py` | PyTorch inference script for the adapted SAM. Performs fully automated, prompt-free segmentation of four orchid floral organs. |
| `NON-SAM model predict.py` | PaddlePaddle inference script for the baseline models (CCNet, DMNet, FCN, UNViT). |
| `requirement.txt` | Python dependencies for both the SAM and non-SAM pipelines. |
| `TEST Images.rar` | Test images covering diverse *Cymbidium* species and floral patterns, for model verification. |

> **Note:** Model weights are hosted outside the repository (see [Model weights](#model-weights) below). Test images and reference labels are being added to this repository and will be updated here.

## Model weights

Pretrained checkpoints are fine-tuned on our orchid floral organ dataset and are distributed as follows:

| Model | Format | Download |
| --- | --- | --- |
| Fine-tuned SAM (OrchidSAM) | `.pth` | [Google Drive](https://drive.google.com/drive/folders/1-obbtQVtFj-BGAIBzLUgEblbkkT8J5eu?usp=sharing) |
| Non-SAM baselines (CCNet, DMNet, FCN, UNViT) | `.pdparams` | [GitHub Releases](https://github.com/wbx1727031/ORCHIDSAM/releases) |

- The fine-tuned SAM weights are shared via Google Drive because of their file size.
- The non-SAM baseline weights are attached as assets to the repository's Releases page.

## Installation

The two pipelines use different deep learning frameworks (PyTorch for SAM, PaddlePaddle for the baselines), so installing them in **separate virtual environments** is recommended.

`requirement.txt` lists the dependencies of both pipelines:

**SAM pipeline**

```
torch>=2.0.0
torchvision>=0.15.0
segment-anything
opencv-python>=4.6.0
Pillow>=9.0.0
scikit-image>=0.19.0
numpy>=1.23.0
pandas>=1.5.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
```

**Non-SAM pipeline**

```
paddlepaddle-gpu>=2.5.0
paddleseg>=2.8.0
numpy>=1.23.0
pandas>=1.5.0
Pillow>=9.0.0
matplotlib>=3.5.0
```

Example setup for the SAM pipeline:

```bash
git clone https://github.com/wbx1727031/ORCHIDSAM.git
cd ORCHIDSAM

conda create -n orchidsam python=3.10 -y
conda activate orchidsam

# install the SAM-pipeline packages listed above
pip install torch torchvision segment-anything opencv-python Pillow scikit-image numpy pandas scikit-learn matplotlib
```

Example setup for the non-SAM pipeline:

```bash
conda create -n orchid-paddle python=3.10 -y
conda activate orchid-paddle

pip install paddlepaddle-gpu paddleseg numpy pandas Pillow matplotlib
```

## Usage

Before running:

1. Download the model weights:
   - **SAM**: download the fine-tuned `.pth` checkpoint from [Google Drive](https://drive.google.com/drive/folders/1-obbtQVtFj-BGAIBzLUgEblbkkT8J5eu?usp=sharing).
   - **Non-SAM baselines**: download the `.pdparams` files from the [Releases](https://github.com/wbx1727031/ORCHIDSAM/releases) page.
2. Extract the test images and reference labels.
3. Set the checkpoint path, input image folder, and output folder in the corresponding script.

### Adapted SAM (prompt-free)

```bash
python "SAM model predict.py"
```

The script segments each input image into the four floral organ types without any manual prompts.

### Non-SAM baselines (CCNet, DMNet, FCN, UNViT)

```bash
python "NON-SAM model predict.py"
```

> **TODO:** confirm the exact script arguments or path variables, the expected input and output formats, and the color mapping of the four organ classes, then fill in this section.

## Output

Predictions are semantic masks for four organ types: sepal, petal, labellum, and gynostemium. The expert-annotated `ReferenceLabels` are provided as RGB masks, so you can compare predictions visually or with standard metrics such as mIoU and F-score.

## Citation

If you use this code, the pretrained weights, or the test data, please cite the paper.

**Paper**

```bibtex
@article{orchidsam,
  title   = {Floral Organ Semantic Segmentation from Chinese Cymbidium Orchid Images via Architecture Adaptation and Fine-tuning of the Segment Anything Model},
  author  = {[Author list]},
  journal = {[Journal]},
  year    = {[Year]}
}
```

## Acknowledgements

This work builds on [Segment Anything](https://github.com/facebookresearch/segment-anything) and [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg).

## License

Please credit the authors when reusing the materials.

## Contact

For questions, please open an issue in this repository or contact **[Name, email]**.
