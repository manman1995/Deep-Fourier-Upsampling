# Deep Fourier Upsampling

## Introduction

This repository is the **official implementation** of the paper, "Deep Fourier Upsampling", where more implementation details are presented.

### 0. class freup_Areadinterpolation()

It corresponds to the implementaion of the Area Interpolation Variant of Deep Fourier Up-sampling



### 1. class freup_Periodicpadding()

It corresponds to the implementaion of the Periodic padding Variant of Deep Fourier Up-sampling




### 2. class freup_Cornerdinterpolation()

It corresponds to the implementaion of the Corner Interpolation Variant of Deep Fourier Up-sampling



## the recommended plug-and-play upsampling operator


### 0. class fresadd()

It means to sum the results from both the spatial up-sampling and the proposed Fourier upsampling.



### 1. class frescat()

It means to concatenate the results from both the spatial up-sampling and the proposed Fourier upsampling.


## Mindspore Version

We also provide the mindspore code at https://github.com/Dragoniss/mindspore-phase2-Deep-Fourier-Upsampling


## Contact

If you have any problem with the released code, please do not hesitate to contact me by email (manman@mail.ustc.edu.cn).

