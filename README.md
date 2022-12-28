# Diverse Perturbations on Feature Space Based Transferable Adversarial Attacks (DFA)

This repository contains the code for the paper: 

**[Diverse Perturbations on Feature Space Based Transferable Adversarial Attacks]**

##Motivation

Transferability of adversarial attacks has recently received much attention, which facilitates adversarial attacks in more practical settings. Nevertheless, existing transferable attacks craft adversarial examples in a deterministic manner, thus trapping into a poor local optimum and significantly degrading the transferability. By contrast, we propose the Diverse-Feature Attack (DFA), which disrupts salient features in a stochastic method to upgrade transferability. More specifically, perturbations in the weighted intermediate features disrupt universal features shared by different models. The weight is obtained by refining aggregate gradient, which adjust the averaged gradient with respect to feature map of the source model. Then, diverse orthogonal initial perturbations disrupt these features in a stochastic manner, searching the space of transferable perturbations more exhaustively to effectively avoid poor local optima. Extensive experiments confirm the superiority of our approach to the existing attacks.

[attack](https://github.com/chengshuyan/DFA/blob/main/Illustration.jpg)

## Requirements

- Python 3.6.3
- Keras 2.2.4
- Tensorflow 1.12.2
- Numpy 1.16.2
- Pillow 4.2.1

## Experiments

#### Introduction

- `attack_delta.py` : the implementation for different attacks.

- `verify-1.py` : the code for evaluating generated adversarial examples on different models.

  You should download the  pretrained models from ( https://github.com/tensorflow/models/tree/master/research/slim,  https://github.com/tensorflow/models/tree/archive/research/adv_imagenet_models) before running the code. Then place these model checkpoint files in `./models_tf`.

#### Example Usage

##### Generate adversarial examples:

- DFA

```
python attack.py --model_name vgg_16 --attack_method DFA --layer_name vgg_16/conv3/conv3_3/Relu --ens 30 --probb 0.9 --output_dir ./adv/DFA/
```

Different attack methods have different parameter setting, and the detailed setting can be found in our paper.

##### Evaluate the attack success rate

```
python verify-1.py --ori_path ./dataset/images/ --adv_path ./adv/FIA/ --output_file ./log.csv
```

## Citing this work

If you find this work is useful in your research, please consider citing:

```
@inproceedings{Cheng2023,
  title={Diverse Perturbations on Feature Space Based Transferable Adversarial Attacks},
  author={Shuyan Cheng, Peng Li, Jianguo Liu, He Xv, Ruchuan Wang},
  year={2023}
  }
```
