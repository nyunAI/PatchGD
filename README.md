<div align="center">

<h2><a href="https://openreview.net/forum?id=6dS1jhdemD/">Pushing the Limits of Gradient Descent for Efficient Learning on Large Images</a></h2>

[Deepak Gupta](https://dkgupta90.github.io/), [Gowreesh Mago](https://scholar.google.com/citations?user=ewFlY0gAAAAJ&hl=en), [Arnav Chavan](https://sites.google.com/view/arnavchavan/), [Dilip Prasad](https://en.uit.no/ansatte/person?p_document_id=615677), [Rajat Mani Thomas](https://scholar.google.nl/citations?user=gw2bllMAAAAJ&hl=en)

</div>

## Accepted in TMLR [OpenReview](https://openreview.net/forum?id=6dS1jhdemD)

### Abstract

Traditional deep learning models are trained and tested on relatively low-resolution images ($<300$ px), and cannot be directly operated on large-scale images due to compute and memory constraints. We propose Patch Gradient Descent (PatchGD), an effective learning strategy that allows us to train the existing CNN and transformer architectures (hereby referred to as deep learning models) on large-scale images in an end-to-end manner. PatchGD is based on the hypothesis that instead of performing gradient-based updates on an entire image at once, it should be possible to achieve a good solution by performing model updates on only small parts of the image at a time, ensuring that the majority of it is covered over the course of iterations. PatchGD thus extensively enjoys better memory and compute efficiency when training models on large-scale images. PatchGD is thoroughly evaluated on PANDA, UltraMNIST, TCGA, and ImageNet datasets with ResNet50, MobileNetV2, ConvNeXtV2, and DeiT models under different memory constraints. Our evaluation clearly shows that PatchGD is much more stable and efficient than the standard gradient-descent method in handling large images, especially when the compute memory is limited.




## Code usage details


### Setup the environment

Create a conda envirnoment:
```bash
conda create -n pgd python=3.12
conda activate pgd
```
Install requirements using the following command: 
```bash
pip install -r requirements.txt
```

### Data
Experiments mentioned in the paper use the following datasets:
<ol>
    <li><a href="https://www.kaggle.com/c/prostate-cancer-grade-assessment/data">Prostate cANcer graDe Assessment (PANDA)</a></li>
    <li><a href="https://www.nature.com/articles/s41597-024-03587-4">UltraMNIST</a></li>
    <li><a href="https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description">ImageNet</a></li>
    <li><a href="https://www.cancer.gov/ccg/research/genome-sequencing/tcga">TCGA</a></li>
</ol>

<a href="https://www.kaggle.com/c/prostate-cancer-grade-assessment/data">PANDA</a> and <a href="https://www.nature.com/articles/s41597-024-03587-4">UltraMNIST</a> dataset processing scripts are included in the [utility_codes](./utility_codes) directory where folds for PANDA and full dataset for UltraMNIST can be generated. 

For ImageNet and TCGA(LUAD and LUSC), the dataset can be downloaded from Kaggle (for ImageNet) and (<a href="https://portal.gdc.cancer.gov/projects/TCGA-LUAD">LUAD</a> & <a href="https://portal.gdc.cancer.gov/projects/TCGA-LUSC">LUSC</a> with setup instructions listed <a href="https://gdc.cancer.gov/access-data/gdc-data-transfer-tool">here</a>) the splits can be made from the dataset.




### File structure
<ul>
    <li><a href="./baselines/">baselines</a> directory contains the code to run baseline experiments mentioned in the paper for PANDA and UltraMNIST</li>
    <li><a href="./HAR_1d_example/">HAR_1d_example</a> directory contains code to run experiments on the <a href="https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones">Human Activity Recognition dataset</a> (1-d generalization of PatchGD)</li>
    <li><a href="./patch_gd/">patch_gd</a> directory contains the code to run experiments using PatchGD algorithm for PANDA and UltraMNIST</li>
    <li><a href="./utility_codes/">utility_codes</a> directory contains utitlity codes for PANDA and UltraMNIST including creating dataset and folds, calculation of stats, running multiple experiments on multiple gpus etc.</li>
</ul>


# Citation

Please cite using the following citation:

```BibTeX
@article{gupta2023patch,
  title={Patch gradient descent: Training neural networks on very large images},
  author={Gupta, Deepak K and Mago, Gowreesh and Chavan, Arnav and Prasad, Dilip K},
  journal={arXiv preprint arXiv:2301.13817},
  year={2023}
}
```
