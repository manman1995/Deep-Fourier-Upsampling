## Applications
###  Image deraining 
####  Prepare data
Download the training data and add the data path to the config file (/basicsr/option/train/RAIN200H(L)/*.yml). 
#### : Training
```
python /basicsr/train.py -opt /LLIE/options/train/RAIN200H/LPNet_corner.yml
python /basicsr/train.py -opt /LLIE/options/train/RAIN200H/LPNet_padding.yml
python /basicsr/train.py -opt /LLIE/options/train/RAIN200H/LPNet_v1.yml
python /basicsr/train.py -opt /LLIE/options/train/RAIN200H/LPNet_v2.yml
python /basicsr/train.py -opt /LLIE/options/train/RAIN200L/LPNet_corner.yml
python /basicsr/train.py -opt /LLIE/options/train/RAIN200L/LPNet_padding.yml
python /basicsr/train.py -opt /LLIE/options/train/RAIN200L/LPNet_v1.yml
python /basicsr/train.py -opt /LLIE/options/train/RAIN200L/LPNet_v2.yml
```
#### : Inference
Download the pretrained image deraining model from [Google Drive](https://drive.google.com/drive/folders/1zayArqjtukQu9HmtkWQlGzynRNRi-idt?usp=sharing
) and add the path to the config file (/LLIE/options/test/RAIN200H(L)/*.yml).
```
python /basicsr/test.py -opt /LLIE/options/train/RAIN200H/LPNet_corner.yml
python /basicsr/test.py -opt /LLIE/options/train/RAIN200H/LPNet_padding.yml
python /basicsr/test.py -opt /LLIE/options/train/RAIN200H/LPNet_v1.yml
python /basicsr/test.py -opt /LLIE/options/train/RAIN200H/LPNet_v2.yml
python /basicsr/test.py -opt /LLIE/options/train/RAIN200L/LPNet_corner.yml
python /basicsr/test.py -opt /LLIE/options/train/RAIN200L/LPNet_padding.yml
python /basicsr/test.py -opt /LLIE/options/train/RAIN200L/LPNet_v1.yml
python /basicsr/test.py -opt /LLIE/options/train/RAIN200L/LPNet_v2.yml
