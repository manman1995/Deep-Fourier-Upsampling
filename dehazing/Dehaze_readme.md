## Applications
### : Image Dehazing
#### : Prepare data
Download the training data and add the data path to the config file ( AODNet/config/config.yaml). Please refer to [RESIDE](https://github.com/BookerDeWitt/MSBDN-DFF) for data download. 

#### : AODNet Training

Set path 'output_dir', 'data' in AODNet/config/config.yaml.
Set upsampling version in Line 354-358 in AODNet/model/final_model.py
```
cd AODNet
python AODNet/train.py

```
#### : AODNet Inference

Set the path of the pretrained model in  Line110.
Set upsampling version in Line 354-358 in AODNet/model/final_model.py
```
python AODNet/test.py
```


#### : MSBDN Training

 Please refer to MSBDN(https://github.com/BookerDeWitt/MSBDN-DFF) for data download. 
Set upsampling version in Line 321-324 in MSBDN\networks\MSBDN-DFF-v1-1.py
```
cd MSBDN
python train.py --dataset path_to_dataset/RESIDE_HDF5_all/ --lr 1e-4 --batchSize 16 --model MSBDN-DFF-v1-1 --name MSBDN-DFF

```
#### : MSBDN Inference
Please refer to [Google drive]([https://github.com/BookerDeWitt/MSBDN-DFF](https://drive.google.com/drive/folders/1zayArqjtukQu9HmtkWQlGzynRNRi-idt?usp=sharing)) for data download. 
```
cd MSBDN
python test.py --checkpoint path_to_pretrained_model
```
