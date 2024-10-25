from utils import *
import torch
import json
from flamby.utils import evaluate_model_on_tests
import os

config_file_path = './configs/fed_heart_disease.json'
with open(config_file_path, 'r') as f:
    config = json.load(f)

result_dir = '../results/flamby/fed_heart_disease'

from flamby.datasets.fed_heart_disease import (
    BATCH_SIZE,
    LR,
    Baseline,
    BaselineLoss,
    metric,
    NUM_CLIENTS,
    get_nb_max_rounds
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

# Federated Learning loop
# 2nd line of code to change to switch to another strategy (feed the FL strategy the right HPs)
args = {
            "training_dataloaders": train_dataloaders,
            "model": m,
            "loss": lossfunc,
            "optimizer_class": torch.optim.SGD,
            "learning_rate": LR,
            "num_updates": 100,
            "nrounds": -1,
            "seed":-1
        }


from flamby.strategies.fed_avg import FedAvg
from flamby.strategies.fed_opt import FedAdam
from flamby.strategies.fed_opt import FedYogi
from flamby.strategies.scaffold import Scaffold
from flamby.strategies.fed_prox import FedProx

base_rounds = get_nb_max_rounds(args["num_updates"])
rounds = [base_rounds * factor for factor in [1, 2, 5, 10]]
tuning_strategies = [FedAvg]

result_dict = {} 

for strat in tuning_strategies:
    for round in rounds:
            print(f"Running {strat.__name__} with {round} rounds")

            args["seed"] = 42
            args["nrounds"] = round
            args["learning_rate"] = config["strategies"][strat.__name__]["learning_rate"]
            args["model"] = Baseline()

            strat_specific_args = {
                arg: config["strategies"][strat.__name__][arg]
                for arg in config["strategies"][strat.__name__]
                if arg not in ["learning_rate", "optimizer_class"]
            }

            # Scaffold does not take seed as an argument in FLamby code
            if strat.__name__ == "Scaffold":
                del args["seed"]

            s = strat(**args, **strat_specific_args)
            trained_model = s.run()[0]

            performance = evaluate_model_on_tests(trained_model, test_dataloaders, metric)
            performance = performance['client_test_0']

            if strat.__name__ not in result_dict:
                result_dict[strat.__name__] = [(round, performance)]
            else:
                result_dict[strat.__name__].append((round, performance))

            print(f"Performance: {performance}")

# save result_dict as json
with open(os.path.join(result_dir, 'fl_strategy_tuning.json'), 'w') as fp:
    json.dump(result_dict, fp)

# Store acc,round for the round that gives best performance
best_round, best_performance = max(result_dict["FedAvg"], key=lambda x: x[1])
print(f"Best #round: {best_round} with performance {best_performance}")

seeds = list(range(42, 47))
strategies = [FedAvg, FedAdam, FedYogi, Scaffold, FedProx]
performances = []

# Run strategies with best round on remaining seeds
for strat in strategies:
    for seed in seeds:
        print(f"Running {strat.__name__} for {best_round} rounds and seed {seed}")

        args["seed"] = seed
        args["nrounds"] = best_round
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
df.to_csv(os.path.join(result_dir, 'fl_strategy_best.csv'), index=False)

