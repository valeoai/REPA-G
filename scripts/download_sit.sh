#!/bin/bash

mkdir -p pretrained
cd pretrained


pip install huggingface_hub

# Authenticate (interactive if not already logged in)
if ! huggingface-cli whoami >/dev/null 2>&1; then
  huggingface-cli login
fi

# Download model with resume support
python3 - <<EOF
from huggingface_hub import snapshot_download

repos = [
    "REPA-E/sit-ldm-e2e-vavae",
    "REPA-E/e2e-vavae",
]

for repo in repos:
    snapshot_download(
        repo_id=repo,
        local_dir=repo.split('/')[-1],
        local_dir_use_symlinks=False,
        resume_download=True
    )
EOF

wget https://huggingface.co/nyu-visionx/SiT-collections/resolve/main/SiT-XL-2-256.pt

wget https://www.dl.dropboxusercontent.com/scl/fi/cxedbs4da5ugjq5wg3zrg/last.pt?rlkey=8otgrdkno0nd89po3dpwngwcc&st=apcc645o&dl=0
mv last.pt?rlkey=8otgrdkno0nd89po3dpwngwcc&st=apcc645o&dl=0 SiT-XL-2-256-REPA.pt

# List downloaded files
ls -lh pretrained

cd ..

echo "Downloaded SiT models checkpoints to pretrained/"