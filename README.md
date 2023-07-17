# README

This codebase contains implementations of several OOD detection methods on top of deep generative models. The [model_zoo](./model_zoo/) directory contains code from the paper ["Diagnosing and Fixing Manifold Overfitting in Deep Generative Models"](https://arxiv.org/abs/2204.07172) accepted to TMLR in July 2022.


## Setup

**Make sure that your python is `3.9`**. Higher versions will conflict with the [chi2comb](https://pypi.org/project/chi2comb/) library for generalized chi-square computation (TODO: fix this dependency).

Install the requirements with pip:

```bash
pip install -r requirements.txt
```


