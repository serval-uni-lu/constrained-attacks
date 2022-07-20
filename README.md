# Constrained attacks
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2112.01156-b31b1b.svg)](https://arxiv.org/abs/2112.01156)

## Description

Constrained attacks is a framework for constraints adversarial examples unified across multiple constraints' domain.
It currently supports a large diversity of constraints (linear and non-linear).
We instantiated our framework with two attacks:
- MoEvA2: a multi-objective genetic based approach
- C-PGD: a gradient based approach extended from PGD (cite) to support domain constraints.

To learn more, check out our paper [A Unified Framework for Adversarial Attack and Defense in Constrained Feature
    Space](https://arxiv.org/abs/2112.01156).

## Installation

### Using pip

```shell
pip install constrained-attacks
```

## Dependencies

constrained-attacks requires:

- python = "~3.8"
- numpy = "^1.22.3"
- joblib = "^1.1.0"
- pymoo = "^0.5.0"
- tqdm = "^4.63.1"
- pandas = "^1.4.1"

Additional optional requirements for C-PGD are:
- tensorflow = "2.8"
- adversarial-robustness-toolbox[tensorflow] = "1.10"

## Examples

You can find a usage example
- for MoEvA2: [tests/attacks/moeva/test_moeva_run.py](tests/attacks/moeva/test_moeva_run.py)
- for C-PGD: [tests/attacks/cpgd/test_pgd_run.py](tests/attacks/cpgd/test_pgd_run.py)
- for the constraints definition: [tests/attacks/moeva/url_constraints.py](tests/attacks/moeva/url_constraints.py).

## Citation

If you have used our framework for research purposes, you can cite our publication by:

BibTex:
```
@article{simonetto2021unified,
  title={A unified framework for adversarial attack and defense in constrained feature space},
  author={Simonetto, Thibault and Dyrmishi, Salijona and Ghamizi, Salah and Cordy, Maxime and Traon, Yves Le},
  journal={arXiv preprint arXiv:2112.01156},
  year={2021}
}
```
