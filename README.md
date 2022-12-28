# DFA

This repository contains the official code for the ICCV 2021 paper:

[Feature Importance-aware Transferable Adversarial Attacks](https://arxiv.org/pdf/2107.14185.pdf)

## Motivation

Transferability of adversarial attacks has recently received much attention, which facilitates adversarial attacks in more practical settings. Nevertheless, existing transferable attacks craft adversarial examples in a deterministic manner, thus trapping into a poor local optimum and significantly degrading the transferability. By contrast, we propose the Diverse-Feature Attack (DFA), which disrupts salient features in a stochastic method to upgrade transferability. More specifically, perturbations in the weighted intermediate features disrupt universal features shared by different models. The weight is obtained by refining aggregate gradient, which adjust the averaged gradient with respect to feature map of the source model. Then, diverse orthogonal initial perturbations disrupt these features in a stochastic manner, searching the space of transferable perturbations more exhaustively to effectively avoid poor local optima. Extensive experiments confirm the superiority of our approach to the existing attacks.

<!---
![progressive_attack](https://github.com/AI-secure/PSBA/blob/master/imgs/progressive_attack.png)
-->
