## Applications
### : De-raining
#### : Prepare data
Download the training data and add the data path to the config file ( /data/derain_nips/options/train/*/Z_PRENet.yml，*dataset). Please refer to [RAIN200H] and [RAIN200L] for data download. 
#### : Training
```
Select the network framework you want to use and delete the suffix corresponding to the network framework in path (/data/derain_nips/basicsr/models/archs/.). For example, if you want to train the Area framework, rename [prenet_nips_arch_Area.py] to [prenet_nips_arch.py]
python /basicsr/train_rain.py -opt /data/derain_nips/options/train/RAIN200H/Z_PRENet.yml
python /basicsr/train_rain.py -opt /data/derain_nips/options/train/RAIN200L/Z_PRENet.yml


```
#### : Inference
Download the pretrained low-light image enhancement model from [Google Drive](连接记得换一下) and add the path to the config file (/data/derain_nips/options/test/*/Z_PRENet.yml，*dataset).
```
python /basicsr/train_rain.py -opt /data/derain_nips/options/test/RAIN200H/Z_PRENet.yml
python /basicsr/train_rain.py -opt /data/derain_nips/options/test/RAIN200L/Z_PRENet.yml
