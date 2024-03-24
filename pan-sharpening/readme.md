# Pan-Sharpening Algorithm
This repository contains an implementation of pan-sharpening algorithm. The algorithm is specified in the provided YAML configuration file (option.yaml/option_pannet.yaml). Users are required to configure specific parameters before running the code.
# Configuration
Before using the code, make sure to configure the following parameters in the config.yaml file:

- log_dir: Path to the directory where log files will be stored.
- data_dir_train: Path to the training data directory.
- data_dir_eval: Path to the evaluation data directory.
- algorithm: Specify the pan-sharpening algorithm to use. Choose from the options available in the "model" folder.
- save_dir: Directory to save the inference results.
- test.model: For inference, modify the model checkpoint for testing. This allows running inference using either python test.py or python py-tra/demo_deep_methods.py.
# Checkpoint Link
Download the pre-trained models and checkpoints from this [Google Drive link](https://drive.google.com/drive/folders/1zayArqjtukQu9HmtkWQlGzynRNRi-idt?usp=sharing
) and place them in the appropriate directories.


# Usage
Inference
For inference, after configuring the parameters, run the following commands:

`
python test.py
`
or 
`
python test_pannet.py
`
then
`
python py-tra/demo_deep_methods.py
`

Training

For training, modify the necessary parameters in the config.yaml file, then run:

`python main.py
`
or
`
python main_pannet.py
`
