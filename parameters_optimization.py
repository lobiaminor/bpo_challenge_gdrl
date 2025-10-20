import os

import optuna
import torch
import numpy as np
import tianshou as ts
from torch.optim.lr_scheduler import ExponentialLR
from tianshou.policy import PPOPolicy
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter

from train_graph import HeteroActorNodeSelection, HeteroCriticNodeSelection, save_best_fn, get_env, \
    preprocess_function

problem = 'consulta'
problem_type = 'regenerated'  # 'original' for the original problem, 'regenerated' for the regenerated problem (only for bpi2017)
running_time = 7 * 24
num_cpu = 1
load_model = False
model_name = "complete"
nr_envs = 1

train_args = {"hidden_dim": 64,
              "lr": 1e-3,  # 1e-3 for default training
              "discount_factor": 1,
              "batch_size": 64,
              "max_batch_size_ppo": 0,  # 3200, 3600 # 0 means step_per_collect amount
              "nr_envs": 1,  # 64
              "max_epoch": 30,
              "buffer_size": 200000,
              "step_per_epoch": 20000,  # 6400, 25600, 128000
              "step_per_collect": 10000,
              #"episode_per_collect": 10,
              "repeat_per_collect": 2
              }
def objective(trial, problem_name=problem):
    # Suggest hyperparameters
    #hidden_dim = trial.suggest_int('hidden_dim', 32, 128)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    #discount_factor = trial.suggest_uniform('discount_factor', 0.9, 0.999)
    batch_size = trial.suggest_int('batch_size', 32, 128)
    #nr_envs = trial.suggest_int('nr_envs', 1, 16)
    max_epoch = trial.suggest_int('max_epoch', 10, 100)
    buffer_size = trial.suggest_int('buffer_size', 100000, 500000)
    step_per_epoch = trial.suggest_int('step_per_epoch', 10000, 50000)
    step_per_collect = trial.suggest_int('step_per_collect', 5000, 20000)
    repeat_per_collect = trial.suggest_int('repeat_per_collect', 1, 5)

    # Update train_args with suggested hyperparameters
    train_args.update({
        "hidden_dim": 64,
        "lr": lr,
        "discount_factor": 1,
        "batch_size": batch_size,
        "nr_envs": 2,
        "max_epoch": max_epoch,
        "buffer_size": buffer_size,
        "step_per_epoch": step_per_epoch,
        "step_per_collect": step_per_collect,
        "repeat_per_collect": repeat_per_collect
    })


    if problem_name == 'toloka':
        n_edges = 138
    elif problem_name == 'fines':
        n_edges = 53
    elif problem_name == 'bpi2017':
        n_edges = 573
    elif problem_name == 'bpi2012':
        n_edges = 219
    elif problem_name == 'bpi2018':
        n_edges = 479  # full dataset: 479, with threshold==3000: 181
    elif problem_name == 'consulta':
        n_edges = 439
    elif problem_name == 'production':
        n_edges = 78
    else:
        raise Exception("Invalid problem name")


    # Set up model, optimizer, and policy
    actor_net = HeteroActorNodeSelection(n_edges=n_edges)
    critic_net = HeteroCriticNodeSelection(n_edges=n_edges)
    optim = torch.optim.Adam(list(actor_net.parameters()) + list(critic_net.parameters()), lr=lr)
    scheduler = ExponentialLR(optim, 0.95)
    policy = PPOPolicy(actor_net, critic_net, optim,
                       discount_factor=1,
                       dist_fn=torch.distributions.categorical.Categorical,
                       deterministic_eval=True,
                       lr_scheduler=scheduler,
                       reward_normalization=False)
    policy.action_type = "discrete"

    # Set up environment and collector
    train_envs = ts.env.DummyVectorEnv([lambda: get_env(problem, running_time, problem_type) for _ in range(nr_envs)])
    collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(buffer_size, nr_envs),
                                  exploration_noise=False, preprocess_fn=preprocess_function)
    collector.reset()

    # Set up test collector
    test_envs = ts.env.DummyVectorEnv([lambda: get_env(problem, running_time, problem_type) for _ in range(nr_envs)])
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=False, preprocess_fn=preprocess_function)

    # Set up logger
    log_path = os.path.join(f"logs/ppo_graph/{problem_name}")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    # Train the model
    trainer = ts.trainer.OnpolicyTrainer(
        policy, collector, max_epoch=max_epoch, step_per_epoch=step_per_epoch,
        step_per_collect=step_per_collect, episode_per_test=50, batch_size=batch_size,
        repeat_per_collect=repeat_per_collect, logger=logger, test_in_train=True, verbose=False, test_collector=test_collector, save_best_fn=save_best_fn)
    result = trainer.run()

    # Return the evaluation metric (e.g., average cycle time)
    return result['best_reward']

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("Best hyperparameters: ", study.best_params)