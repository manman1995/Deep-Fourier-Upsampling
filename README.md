###  Image deraining using LP-Net and Deep Fourier Upsampling

#### : Training
```
python basicsr/train.py -opt options/train/RAIN200H/LPNet_corner.yml
python basicsr/train.py -opt options/train/RAIN200H/LPNet_padding.yml
```
#### : Inference
```
python basicsr/test.py -opt options/test/RAIN200H/LPNet_corner.yml
python basicsr/test.py -opt options/test/RAIN200H/LPNet_padding.yml 
python basicsr/test.py -opt options/test/RAIN200H/LPNet_v1.yml 
python basicsr/test.py -opt options/test/RAIN200H/LPNet_v2.yml 

#### : common used cmds
request GPU node:
salloc --account eecs568s001w25_class --partition gpu_mig40,gpu,spgpu 
--nodes 1 --ntasks 1 --cpus-per-task 1 --gpus 1 --mem 16G --time 00:30:00


如果实现不加入conv的周期填充？
为什么要加入conv？
因为：
1. 有很多高频噪声、不规则结构；
2. 直接做周期填充可能会引入spectral artifacts）

需要对一张image进行upsampling,
1. spatial domain 直接upsampling 
2. 做ft得到frequency domain, periodic padding, ift get the spatial domain

in theory, method 1 is equal to method 2. However, 
