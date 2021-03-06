# realistic-ssl-evaluation-pytorch
This repository is reimplementation of [Realistic Evaluation of Deep Semi-Supervised Learning Algorithms](https://arxiv.org/abs/1804.09170), by Avital Oliver*, Augustus Odena*, Colin Raffel*, Ekin D. Cubuk, and Ian J. Goodfellow, arXiv preprint arXiv:1804.09170.

Original repo is [here](https://github.com/brain-research/realistic-ssl-evaluation).

# Requirements
- Python 3.6+
- PyTorch 1.1.0
- torchvision 0.3.0
- numpy 1.16.2

# How to run
Prepare dataset

```python
python build_dataset.py
```

Default setting is SVHN 1000 labels. If you try other settings, please check the options first by ```python build_dataset.py -h```.

Running experiments

```python
python train.py
```

Default setting is VAT. Please check the options by ```python python train.py -h```

# Performance
WIP

|algorithm|paper||this repo| |
|--|--|--|--|--|
||cifar10 4k labels|svhn 1k labels|cifar10 4k labels|svhn 1k labels|
|Supervised|20.26 ±0.38|12.83 ±0.47|20.35±0.14|12.33±0.25
|Pi-Model|16.37 ±0.63|7.19 ±0.27|16.24±0.38|7.81±0.39
|Mean Teacher|15.87 ±0.28|5.65 ±0.47|15.77±0.22|6.48±0.44
|VAT|13.86 ±0.27|5.63 ±0.20|13.83±0.49|5.84±0.20
|VAT+EM|13.13 ±0.39|5.35 ±0.19|13.30±0.27|5.76±0.13
|Pseudo-Label|17.78 ±0.57|7.62 ±0.29|N/A|N/A
|[ICT](https://arxiv.org/abs/1903.03825)|( 7.66 ±0.17 )|( 3.53 ±0.07 )|N/A|N/A
|[MixMatch](https://arxiv.org/abs/1905.02249)|( 6.50 )|( 3.27 ±0.31 )|N/A|N/A

*NOTE: Experimental setting of ICT and MixMatch papers is different from this benchmark.*

# Reference
- [Realistic Evaluation of Deep Semi-Supervised Learning Algorithms](https://arxiv.org/abs/1804.09170), by Avital Oliver*, Augustus Odena*, Colin Raffel*, Ekin D. Cubuk, and Ian J. Goodfellow, arXiv preprint arXiv:1804.09170.
- [Interpolation Consistency Training for Semi-Supervised Learning](https://arxiv.org/abs/1903.03825), by Vikas Verma, Alex Lamb, Juho Kannala, Yoshua Bengio, David Lopez-Paz
- [MixMatch: A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249), by David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver, Colin Raffel
