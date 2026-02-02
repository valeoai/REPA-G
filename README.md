<h1 align="center"> REPA-G - Official implementation of "Test-Time Conditioning with Representation-Aligned Visual Features"</h1>

<p align="center">
  <a href="https://scholar.google.com/citations?user=9Mr--hUAAAAJ" target="_blank">Nicolas&nbsp;Sereyjol-Garros</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://ellingtonkirby.github.io/" target="_blank">Ellington&nbsp;Kirby</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=YhTdZh8AAAAJ&hl=en" target="_blank">Victor&nbsp;Letzelter</a><sup>1,2</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://nerminsamet.github.io/" target="_blank">Nermin&nbsp;Samet</a><sup>1</sup>&ensp;<b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=n_C2h-QAAAAJ&hl=en" target="_blank">Victor&nbsp;Besnier</a><sup>1</sup> &ensp; 
</p>

<p align="center">
  <sup>1</sup> Valeo.ai, Paris, France &emsp; </sub> <sup>2</sup> LTCI, Télécom Paris, Institut Polytechnique de Paris, France  &emsp;
</p>

<p align="center">
  <!-- <a href="">🌐 Project Page</a> &ensp; -->
  <a href="">📃 Paper</a>
</p>

![](assets/teaser.png)

## Overview

![](assets/visuals.png)



## 📚 Citation
If you find our work useful, please consider citing:

```bibtex
@misc{r3dpa,
      title={Leveraging 3D Representation Alignment and RGB Pretrained Priors for LiDAR Scene Generation}, 
      author={Nicolas Sereyjol-Garros and Ellington Kirby and Victor Besnier and Nermin Samet},
      year={2026},
      eprint={2601.07692},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.07692}, 
}
```
## Getting Started
### 1. Environment Setup
To set up our environment, please run:

```bash

```

### 2. Prepare the training data
Download and extract the training split of the [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/index) dataset. Once it's ready, run the following command to preprocess the dataset:

```bash
python preprocessing.py --imagenet-path /PATH/TO/IMAGENET_TRAIN
```

Replace `/PATH/TO/IMAGENET_TRAIN` with the actual path to the extracted training images.


### 3. Download the pretrained model

To Download together REPA-E, REPA and SiT without alignmnet, run the script 

```bash 
bash scripts/download_sit.sh
```

### 4. Demo

streamlit

### 5. (Optional) Download additional visual backbone for evaluation

For evaluation of alignment with anchors with additional image backbone, download the image backbones needed and put them in `ckpts`

* `mocov3` : [this link](https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar) and place it as `./ckpts/mocov3_vitb.pth` 
* `JEPA` : [this link](https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.h.14-900e.pth.tar) and place it as `./ckpts/ijepa_vith.pth`
* `MAE` : [this link](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth) and place it as `./ckpts/mae_vitl.pth`

or run the script 

```bash 
bash scripts/download_image_backbone.sh
```



### 5. Evaluate

To generate samples and save them in a `.npz` file for evaluation, run the following script after after making sure the parameters match your model path. 
```bash
bash scripts/sample.sh 
```


## Acknowledgement
This codebase is largely built upon:
- [REPA-E](https://github.com/End2End-Diffusion/REPA-E)

We sincerely thank the authors for making their work publicly available.

