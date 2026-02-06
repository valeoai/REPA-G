#!/bin/bash

export TORCH_HOME=$HOME/.cache/torch

torchrun --nproc_per_node=1 generate.py \
    --data-dir data/ImageNet \
    --mode sde \
    --num-steps 250 \
    --cfg-scale 1.0 \
    --anchor-seed 42 \
    --num-fid-samples 50000 \
    --pproc-batch-size 64 \
    --label-sampling equal \
    --compute-similarity \
    --repa-lambda 50000 \
    --compute-conditioning-alignment \
    --gibbs \
    --sample-dir log/samples \
    --additional-similarity-backbones "dinov2-vit-b,clip-vit-L,mocov3-vit-b,jepa-vit-h,mae-vit-l"\
    --compute-metrics \
    --ref-batch log/samples/VIRTUAL_imagenet256_labeled.npz \
    --use-mlflow \
    --mlflow-tracking-uri log/mlruns \
    --use-feature-conditioning \
    --exp-path pretrained/SiT-XL-2-256-REPA.pt \
    --feature-type=sit \
    --average-features \
    --use-uncond-class \