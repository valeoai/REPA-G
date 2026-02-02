# Code for Toy example

## Installation

Install torch compatible with you compute platform, then run

```bash
pip install -r experiments.txt
```

## Content

`dataset.py` includes the definition of the dataset and the mapping to the feature space.

`model.py` provides the formulation of the potential and the diffusion mlp with the sampling procedure to generate with or without feature conditioning.

`train.py` contains the training recipee to train the diffusion model with alignment.

To visualize results, run `toy_example.ipynb`.




