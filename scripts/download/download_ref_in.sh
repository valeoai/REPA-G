#!/bin/bash

mkdir -p log/samples
cd log/samples

wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz

cd ../..