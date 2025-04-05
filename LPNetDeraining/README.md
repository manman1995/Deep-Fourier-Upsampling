# LPNetDeraining

## 1. Introduction
This folder contains the implementation of the LPNet model for image deraining using deep Fourier upsampling techniques. The project leverages Theorem-1 in Deep Fourier Upsampling,"periodic padding", to enhance image quality by effectively removing rain streaks from images.

## 2. Training and Inference Commands
### Training
To train the model, use the following commands:
```bash
python basicsr/train.py -opt options/train/pad.yml
python basicsr/train.py -opt options/train/pad_area.yml
python basicsr/train.py -opt options/train/pad_corner.yml
python basicsr/train.py -opt options/train/pad_larger_kernel.yml
python basicsr/train.py -opt options/train/pad_theory.yml
python basicsr/train.py -opt options/train/pad_attention.yml


```

### Inference
To test the model on a dataset, use:
```bash
python basicsr/train.py -opt options/test/pad.yml
python basicsr/train.py -opt options/test/pad_area.yml
python basicsr/train.py -opt options/test/pad_corner.yml
python basicsr/train.py -opt options/test/pad_larger_kernel.yml
python basicsr/train.py -opt options/test/pad_theory.yml
python basicsr/train.py -opt options/test/pad_attention.yml
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

