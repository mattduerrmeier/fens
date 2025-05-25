# FENS-VAE

This repository contains the code for our project FENS-VAE, an application of FENS to Variational Autoencoders (VAE).
It is based on [FENS code repository](https://github.com/sacs-epfl/fens).

FENS paper: **"Revisiting Ensembling in One-Shot Federated Learning", NeurIPS 2024**. [(link)](https://proceedings.neurips.cc/paper_files/paper/2024/file/7ea46207ec9bda974b140fe11d8dd727-Paper-Conference.pdf)

## Installation

This project requires Python 3.9.
The dependencies can be installed using conda or pipenv.

```bash
# to install via conda; will require to install flamby from source
conda create -n fens python=3.9 -f environment.yml
conda activate fens

# to install via pipenv
pipenv install
```

## Wandb

You can optionally use [wandb](https://wandb.ai/) for logging the local training and aggregation results.
You need to create an account on wandb to use this feature.
Once you have an account, use the following command to login:

```bash
wandb login
```

You must also add these two parameters to the scripts in `scripts/flamby/` to run with Wandb:

```bash
# remove --disable_wandb
# add these two parameters instead:
--wandb_project <project-name> --wandb_entity <entity-name>
```

## FENS-VAE FLamby Setup

> [!IMPORTANT]
> you need to install FLamby from the source to reproduce these experiments. Fens relies on it to load the datasets, and they must be installed via FLamby. If you don't install them this way, it will throw an error that the datasets are not available.

FLamby's preprocessing and loading components are available on [GitHub](https://github.com/owkin/FLamby).
FENS requires that you install the dataset through FLamby.
To ensure that all developers working on this project use the same version of FLamby, we added the FLamby repository as a submodule to this repository.

This project's dependency is properly defined in the `Pipfile`.
If you use `pipenv` to manage your dependencies, FLamby will be properly built when installing this project's dependencies.

If you don't use pipenv, you can run the following commands to get FLamby properly set up.
Since FLamby is not published as a PyPI package, you need to install it from source.

```bash
cd flamby-repo
pip install -e .[heart,isic2019] # list of the datasets you want to install
```

Regardless of the installation method, you still need to download the datasets via FLamby.
Visit its repository for installation instructions.
- [Fed-heart instructions](./flamby-repo/flamby/datasets/fed_heart_disease/README.md)
- [Fed-ISIC instructions](./flamby-repo/flamby/datasets/fed_isic2019/README.md)

## Fens FLamby Experiments

If not already done, please follow the instructions in the Flamby repository to setup the python environment and the two datasets, `Fed-ISIC2019` and `Fed-Heart-Disease`.
After the setup and activation of the conda environment, the following line should execute without errors.

```python
from flamby.datasets.fed_isic2019 import (
            BATCH_SIZE,
            LR,
            NUM_EPOCHS_POOLED,
            NUM_CLIENTS,
            Optimizer,
            Baseline,
            BaselineLoss,
            FedIsic2019 as FedDataset
        )
```

Then, you can run the evaluation with the following script:

```bash
# flamby/args.py a list of possible arguments
# we provide scripts for each of the datasets evaluated
./scripts/flamby/mnist_local_training.sh
```
