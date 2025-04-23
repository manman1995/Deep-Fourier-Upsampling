
# Deep Fourier Upsampling

## Introduction
The **Deep Fourier Upsampling** project explores advanced image processing 
techniques using Fourier transforms. The primary goal is to enhance image 
quality through innovative methods such as periodic padding and deep 
learning-based architectures. This repository includes implementations for 
tasks like image deraining, super-resolution, and other image restoration 
problems.

### Key Components
1. **LPNetDeraining**:
   - Implements the LPNet model for removing rain streaks from images.
   - Leverages Theorem-1 from *Deep Fourier Upsampling* for periodic padding.
   - Includes training and inference pipelines with configurable options.

2. **Datasets**:
   - Contains datasets for training and testing, such as Rain100H and 
     RainTrainH.
   - Organized into ground truth and input (rainy) image directories.

3. **Pretrained Models**:
   - Provides pretrained weights for various configurations of LPNet.

4. **Configurations**:
   - YAML files for training and testing, specifying paths, network 
     architectures, and hyperparameters.

5. **Scripts**:
   - `train.py`: For training models.
   - `test.py`: For testing models on datasets.

### Authors
- Yiwei Gui
- Jiadong Hao
- Yiting Wang
- Hefeng Zhou
