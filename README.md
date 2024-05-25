# Improving the Transferability of Adversarial Attacks via Self-Ensemble

This repository contains the code for the paper: 

**[Improving the Transferability of Adversarial Attacks via Self-Ensemble](https:)**

## Requirements

- Python 3.6.3
- Keras 2.2.4
- Tensorflow 1.12.2
- Numpy 1.16.2
- Pillow 4.2.1

## Experiments

#### Introduction

- `attack.py` : the implementation for different attacks.

- `verify.py` : the code for evaluating generated adversarial examples on different models.

  You should download the  pretrained models from ( https://github.com/tensorflow/models/tree/master/research/slim,  https://github.com/tensorflow/models/tree/archive/research/adv_imagenet_models) before running the code. Then place these model checkpoint files in `./models_tf`.

#### Example Usage

##### Generate adversarial examples:

- SEFA

```
python attack.py --model_name vgg_16 --attack_method FIA --layer_name vgg_16/conv3/conv3_3/Relu --ens 30 --probb 0.7 --output_dir ./adv/FIA/
```

- PIM:

```
python attack.py --model_name vgg_16 --attack_method PIM --amplification_factor 10 --gamma 1 --Pkern_size 3 --output_dir ./adv/PIM/
```

- SEFA+PIDIM

```
python attack.py --model_name vgg_16 --attack_method FIAPIDIM --layer_name vgg_16/conv3/conv3_3/Relu --ens 30 --probb 0.7 --amplification_factor 2.5 --gamma 0.5 --Pkern_size 3 --image_size 224 --image_resize 250 --prob 0.7 --output_dir ./adv/FIAPIDIM/
```

Different attack methods have different parameter setting, and the detailed setting can be found in our paper.

##### Evaluate the attack success rate

```
python verify.py --ori_path ./dataset/images/ --adv_path ./adv/FIA/ --output_file ./log.csv
```

## Citing this work

If you find this work is useful in your research, please consider citing:

```
@inproceedings{cheng,
  title={Improving the Transferability of Adversarial Attacks via Self-Ensemble},
  author={Shuyan Cheng, Peng Li, Jianguo Liu, He Xu, and Yudong Yao},
  booktitle={},
  pages={},
  year={2024}
  }
```
