# LPNetDeraining

## 1. Introduction
This folder contains the implementation of the **LPNet** model for image 
deraining using deep Fourier upsampling techniques. The project leverages 
Theorem-1 from *Deep Fourier Upsampling*, specifically **periodic padding**, 
to enhance image quality by effectively removing rain streaks from images.

## 2. Training and Inference Commands
### Training
To train the model, ensure the following fields are correctly set in the config 
(e.g., `pad_area.yml`):

```yaml
# Ground truth and input (rainy) image paths
dataroot_gt: /home/haojd/Fourier/deraining/lpnet/datasets/RainTrainH/norain
dataroot_lq: /home/haojd/Fourier/deraining/lpnet/datasets/RainTrainH/rain

# Validation image path
dataroot_gt: /home/haojd/Fourier/deraining/lpnet/datasets/Rain100H
dataroot_lq: /home/haojd/Fourier/deraining/lpnet/datasets/Rain100H/rainy

# Network architecture (must match the training config)
network_g:
  type: LPNet_combined_area_pad
```

Then run training with:

```bash
python basicsr/train.py -opt options/train/pad_area.yml
```

### Inference
To test the model, ensure the following fields are correctly set in the config 
(e.g., `pad_area.yml`):

```yaml
# Ground truth and input (rainy) image paths
dataroot_gt: /scratch/eecs568s001w25_class_root/eecs568s001w25_class/yiweigui/\
Deep-Fourier-Upsampling/Dataset/Rain100H/norain
dataroot_lq: /scratch/eecs568s001w25_class_root/eecs568s001w25_class/yiweigui/\
Deep-Fourier-Upsampling/Dataset/Rain100H/rainy

# Pretrained model path
path:
  pretrain_network_g: /scratch/eecs568s001w25_class_root/eecs568s001w25_class/\
yiweigui/Deep-Fourier-Upsampling/Pretrained_model/pad_fusion_v2/net_g_4500.pth

# Network architecture (must match the training config)
network_g:
  type: LPNet_combined_area_pad
```

Then run inference with:

```bash
python basicsr/test.py -opt options/test/pad_area.yml
```

## 3. Commonly Used Command-Line Instructions
- **Request a GPU node**:
  ```bash
  salloc --account eecs568s001w25_class --partition gpu_mig40,gpu,spgpu \
  --nodes 1 --ntasks 1 --cpus-per-task 1 --gpus 1 --mem 16G --time 00:30:00
  ```
- **Check job queue**:
  ```bash
  squeue -A eecs568s001w25_class
  ```
- **Cancel a job**:
  ```bash
  scancel <job ID>
  ```

## 4. Authors
- Yiwei Gui
- Jiadong Hao
- Yiting Wang
- Hefeng Zhou

