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
scancel <job ID>

squeue -A eecs568s001w25_class

[yiweigui@gl-login4 Plot]$ python parse_log_plot_psnr.py pad_attention.log pad_fusion_v1.log pad_fusion_v2.log pad_larger_kernel.log pad_theory.log pad.log --labels "pad_attention" "pad_fusion_v1" "pad_fusion_v2" "pad_larger_kernel" "pad_theory" "pad" --output psnr_plot.png --title "PSNR Comparison" --scale_flags 1 1 1 1 1 0.98 --psnr_info_output default

如果实现不加入conv的周期填充？
为什么要加入conv？
因为：
1. 有很多高频噪声、不规则结构；
2. 直接做周期填充可能会引入spectral artifacts）

需要对一张image进行upsampling,
1. spatial domain 直接upsampling 
2. 做ft得到frequency domain, periodic padding, ift get the spatial domain

in theory, method 1 is equal to method 2. However, 

Implementation of theorem-1
1. add Deraining_LPNET/basicsr/models/archs/LPNet_pad_theory_arch.py


Current Problems:
cannot both train lpnet_pad and lpnet_pad_theory model. It seems be related to 
loading the pretrained model. (*.pth files) 

Solution：
检查Deraining_LPNET/experiments/RAIN200H-LPNet_pad/training_states，是否有state
文件，他会自动恢复训练。如果想从头训练，需要删除。 

Result: 
padding: 2025-04-03 18:32:34,678 INFO: Validation RAIN200H,  # psnr: 16.7730
padding_theory: 2025-04-03 18:37:01,698 INFO: Validation RAIN200H,
    # psnr: 17.2927

Analysis: the result is counter-intuitive, since the padding_theory has better
    performance. However, this is just a very little comparsion and datasets are
    very tiny (100 pictures). I will try introduce more detailed config and 
    larger dataset in this comparsion.

i love  me