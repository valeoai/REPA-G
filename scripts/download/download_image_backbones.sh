#!/bin/bash

mkdir -p ckpts
cd ckpts

wget https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar
mv vit-b-300ep.pth.tar mocov3_vitb.pth

wget https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.h.14-900e.pth.tar
mv IN22K-vit.h.14-900e.pth.tar ijepa_vith.pth

wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth
mv mae_pretrain_vit_large.pth mae_vitl.pth

cd ..

echo "Downloaded image backbone checkpoints to ckpts/"