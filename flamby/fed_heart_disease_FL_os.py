from utils import *
import torch
import json
from flamby.utils import evaluate_model_on_tests
import os
import sys

# get num_updates from command line
if len(sys.argv) < 2:
    print('Please provide number of updates a.k.a local steps. Exiting...')
    exit(0)
num_updates = int(sys.argv[1])

config_file_path = './configs/fed_heart_disease.json'
with open(config_file_path, 'r') as f:
    config = json.load(f)

result_dir = '../results/flamby/fed_heart_disease_os/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

from flamby.datasets.fed_heart_disease import (
    BATCH_SIZE,
    LR,
    Baseline,
    BaselineLoss,
    metric,
    NUM_CLIENTS
)
from flamby.datasets.fed_heart_disease import FedHeartDisease as FedDataset


# We loop on all the clients of the distributed dataset and instantiate associated data loaders
train_dataloaders = [
            torch.utils.data.DataLoader(
                FedDataset(center = i, train = True, pooled = False),
                batch_size = BATCH_SIZE,
                shuffle = True,
                num_workers = 0
            )
            for i in range(NUM_CLIENTS)
        ]

# We only instantiate one test set in this particular case: the pooled one
test_dataloaders = [
            torch.utils.data.DataLoader(
                FedDataset(train = False, pooled = True),
                batch_size = BATCH_SIZE,
                shuffle = False,
                num_workers = 0,
            )
        ]

lossfunc = BaselineLoss()
m = Baseline()

from flamby.strategies.fed_avg import FedAvg
from flamby.strategies.fed_opt import FedAdam
from flamby.strategies.fed_opt import FedYogi
from flamby.strategies.scaffold import Scaffold
from flamby.strategies.fed_prox import FedProx

# Store acc,round for the round that gives best performance
args = {
            "training_dataloaders": train_dataloaders,
            "model": m,
            "loss": lossfunc,
            "optimizer_class": torch.optim.SGD,
            "learning_rate": LR,
            "nrounds": 1,
            "num_updates": -1,
            "seed":-1
        }

seeds = list(range(42, 47))

strategies = [FedAvg, FedAdam, FedYogi, Scaffold, FedProx]
performances = []

# Run strategies with best round on remaining seeds
for strat in strategies:
    for seed in seeds:
        print(f"Running {strat.__name__} for 1 round, {num_updates} local steps and seed {seed}")

        args["seed"] = seed
        args["num_updates"] = num_updates
        args["learning_rate"] = config["strategies"][strat.__name__]["learning_rate"]
        args["model"] = Baseline()

        strat_specific_args = {
            arg: config["strategies"][strat.__name__][arg]
            for arg in config["strategies"][strat.__name__]
            if arg not in ["learning_rate", "optimizer_class"]
        }

        if strat.__name__ == "Scaffold":
            del args["seed"]

        s = strat(**args, **strat_specific_args)
        trained_model = s.run()[0]

        performance = evaluate_model_on_tests(trained_model, test_dataloaders, metric)
        performance = performance['client_test_0']

        performances.append([strat.__name__, seed, performance])

        print(f"Performance: {performance}")

# Convert performances into a dataframe
import pandas as pd
df = pd.DataFrame(performances, columns=['strategy', 'seed', 'performance'])
df.to_csv(os.path.join(result_dir, f'fl_strategy_os_{num_updates}.csv'), index=False)

