#!/bin/bash
# SBATCH --account=eecs568s001w25_class
# SBATCH --partition=spgpu
# SBATCH --nodes=1
# SBATCH --ntasks=1
# SBATCH --cpus-per-task=10
# SBATCH --gpus=1
# SBATCH --mem=60G
# SBATCH --time=4:00:00
# SBATCH --job-name=deeplab_train
# SBATCH --output=logs/train_%j.log   # 日志输出路径（%j 是 job id）

# === 检查 GPU 状态 ===
nvidia-smi

# === 加载环境 ===
source ~/.bashrc
conda activate deeplab

# === 切换到你的项目目录 ===
cd /home/zhouhf/EECS556_25WN/DeepLabV3Plus-Pytorch

# === 启动训练 ===
python main.py --model deeplabv3plus_mobilenet --gpu_id 0 --year 2012_aug --crop_val --lr 0.02 --crop_size 513 --batch_size 8 --output_stride 16

