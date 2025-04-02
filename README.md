###  Image deraining
####  Prepare data
Download the training data and add the data path to the config file (/basicsr/option/train/RAIN200H(L)/*.yml). 
#### : Training
```

```
#### : Inference
Download the pretrained image deraining model from [Google Drive](https://drive.google.com/drive/folders/1zayArqjtukQu9HmtkWQlGzynRNRi-idt?usp=sharing
) and add the path to the config file (Deraining_LPNET/options/test/RAIN200H/*.yml).
```
python basicsr/test.py -opt options/test/RAIN200H/LPNet_corner.yml
python basicsr/test.py -opt options/test/RAIN200H/LPNet_padding.yml 
python basicsr/test.py -opt options/test/RAIN200H/LPNet_v1.yml 
python basicsr/test.py -opt options/test/RAIN200H/LPNet_v2.yml 
