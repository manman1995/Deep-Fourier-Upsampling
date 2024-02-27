## Applications
###  Low-Light Image Enhancement
####  Prepare data
Download the training data and add the data path to the config file (/basicsr/option/train/LLIE/*.yml). Please refer to [LOL](https://daooshee.github.io/BMVC2018website/) and [Huawei](https://github.com/JianghaiSCU/R2RNet) for data download. 
#### : Training
```
python /LLIE/train.py -opt /LLIE/options/train/LLIE/SID_UpSampling_Padding.yml
python /LLIE/train.py -opt /LLIE/options/train/LLIE/SID_UpSampling_Area.yml
python /LLIE/train.py -opt /LLIE/options/train/LLIE/SID_UpSampling_AreaV2.yml
python /LLIE/train.py -opt /LLIE/options/train/LLIE/SID_UpSampling_Corner.yml
python /LLIE/train.py -opt /LLIE/options/train/LLIE/DRBN_UpSampling_Padding.yml
python /LLIE/train.py -opt /LLIE/options/train/LLIE/DRBN_UpSampling_Area.yml
python /LLIE/train.py -opt /LLIE/options/train/LLIE/DRBN_UpSampling_AreaV2.yml
python /LLIE/train.py -opt /LLIE/options/train/LLIE/DRBN_UpSampling_Corner.yml
```
#### : Inference
Download the pretrained low-light image enhancement model from [Google Drive](https://drive.google.com/drive/folders/1zayArqjtukQu9HmtkWQlGzynRNRi-idt?usp=sharing
) and add the path to the config file (/LLIE/options/test/LLIE/*.yml).
```
python /LLIE/train.py -opt /LLIE/options/train/LLIE/SID.yml
python /LLIE/train.py -opt /LLIE/options/train/LLIE/DRBN.yml
