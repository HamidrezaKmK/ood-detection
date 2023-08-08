# README

This repository explores different ways of performing OOD detection on likelihood-based deep generative models. The codebase uses [two_step_zoo](https://github.com/layer6ai/two_step_zoo), containing code from the paper ["Diagnosing and Fixing Manifold Overfitting in Deep Generative Models"](https://arxiv.org/abs/2204.07172) accepted to TMLR in July 2022.


## Setup

**Make sure that your python is `3.9`**. Higher versions will conflict with the [chi2comb](https://pypi.org/project/chi2comb/) library for generalized chi-square computation (TODO: fix this dependency).

Install the requirements with pip:

```bash
pip install -r requirements.txt
```

## Environment Variables

You should set your environment variables for the root directory of data and the root directory of models with the following:

```
dotenv set MODEL_DIR <root-path-to-model-configurations-and-weights>
dotenv set DATA_DIR <root-path-to-data-directory>
```
