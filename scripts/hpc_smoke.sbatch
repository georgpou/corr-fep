#!/usr/bin/env bash
#SBATCH --job-name=corr-fep-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=corr-fep-smoke-%j.out
# Site-specific: add --partition, --account, --gres=gpu:1 or --gpus=1 if needed.
set -euo pipefail
python -m corr_fep.cli doctor
python -m corr_fep.cli demo --seed 17
python -m pytest -q
