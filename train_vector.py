import os
import torch

import numpy as np
import tianshou as ts

from torch import nn, softmax
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import HeteroData
from tianshou.utils import TensorboardLogger
from tianshou.policy import PPOPolicy
from torch.optim.lr_scheduler import ExponentialLR, LinearLR
import torch.nn.functional as F

from bpo_env_graph import BPOEnv

# possible problems are: fines, bpi2017, bpi2018
problem = 'microsoft'
problem_type = 'regenerated'  # 'original' for the original problem, 'regenerated' for the regenerated problem (only for bpi2017)
running_time = 7 * 24
num_cpu = 1
load_model = False
model_name = "complete"
interarrival_rate_multiplier = 1  # a value between 0 and 1 to control the arrival rate of cases



train_args = {"hidden_dim": 64,
              "lr": 1e-3,  # 1e-3 for default training
              "discount_factor": 1,
              "batch_size": 64,
              "max_batch_size_ppo": 0,  # 3200, 3600 # 0 means step_per_collect amount
              "nr_envs": 2,  # 64
              "max_epoch": 100,
              "buffer_size": 200000,
              "step_per_epoch": 20000,  # 6400, 25600, 128000
              "step_per_collect": 10000,
              #"episode_per_collect": 10,
              "repeat_per_collect": 2
              }

def tensor_to_numpy(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    elif isinstance(obj, dict) or isinstance(obj, ts.data.Batch):
        return {key: tensor_to_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, np.ndarray):
        return np.array([tensor_to_numpy(item) for item in obj])
    else:
        return obj

def preprocess_function(normalize=True, **kwargs):
    def normalize_features(x):
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        return (x - mean) / (std + 1e-6)

    if normalize:
        if "obs" in kwargs:
            obs_with_tensors = [
                {"resources": normalize_features(torch.from_numpy(obs['resources']).float()),
                 "task_types": normalize_features(torch.from_numpy(obs["task_types"]).float()),
                 "assignments": normalize_features(torch.from_numpy(obs["assignments"]).float()),
                 "reconstruct_edges": torch.from_numpy(obs["reconstruct_edges"]).long(),
                 "mask": torch.from_numpy(obs["mask"]).bool()}
                for obs in kwargs["obs"]]
            kwargs["obs"] = obs_with_tensors

        if "obs_next" in kwargs:
            obs_with_tensors = [
                {"resources": normalize_features(torch.from_numpy(obs['resources']).float()),
                 "task_types": normalize_features(torch.from_numpy(obs["task_types"]).float()),
                 "assignments": normalize_features(torch.from_numpy(obs["assignments"]).float()),
                 "reconstruct_edges": torch.from_numpy(obs["reconstruct_edges"]).long(),
                 "mask": torch.from_numpy(obs["mask"]).bool()}
                for obs in kwargs["obs_next"]]
            kwargs["obs_next"] = obs_with_tensors
    else:
        if "obs" in kwargs:
            obs_with_tensors = [
                {"resources": torch.from_numpy(obs['resources']).float(),
                 "task_types": torch.from_numpy(obs["task_types"]).float(),
                 "assignments": torch.from_numpy(obs["assignments"]).float(),
                 "reconstruct_edges": torch.from_numpy(obs["reconstruct_edges"]).long(),
                 "mask": torch.from_numpy(obs["mask"]).bool()}
                for obs in kwargs["obs"]]
            kwargs["obs"] = obs_with_tensors

        if "obs_next" in kwargs:
            obs_with_tensors = [
                {"resources": torch.from_numpy(obs['resources']).float(),
                 "task_types": torch.from_numpy(obs["task_types"]).float(),
                 "assignments": torch.from_numpy(obs["assignments"]).float(),
                 "reconstruct_edges": torch.from_numpy(obs["reconstruct_edges"]).long(),
                 "mask": torch.from_numpy(obs["mask"]).bool()}
                for obs in kwargs["obs_next"]]
            kwargs["obs_next"] = obs_with_tensors

    return kwargs

def save_best_fn(policy):
    print("Saving new best policy")
    torch.save(policy.state_dict(), f"ppo_graph_{problem}.pt")

def get_env(problem_name, duration, problem_type='original'):
    if problem_name == 'toloka':
        instance_file = "./data/toloka_problem.pkl"
    elif problem_name == 'fines':
        instance_file = "./data/fines_problem.pkl"
    elif problem_name == 'bpi2017':
        del train_args["step_per_collect"]
        train_args["episode_per_collect"] = 10
        if problem_type == 'original':
            instance_file = "./BPI Challenge 2017 - instance.pickle"
        else:
            instance_file = "./data/bpi2017_problem.pkl"
    elif problem_name == 'bpi2012':
        instance_file = "./data/bpi2012_problem.pkl"
    elif problem_name == 'bpi2018':
        instance_file = "./data/bpi2018_problem.pkl"
    elif problem_name == 'consulta':
        instance_file = "./data/consulta.pkl"
    elif problem_name == 'production':
        instance_file = "data/production.pkl"
    elif problem_name == 'microsoft':
        instance_file = "./data/microsoft.pkl"
    else:
        raise Exception("Invalid problem name")

    env = BPOEnv(instance_file=instance_file,
                 running_time=duration, action_mode='edge_selection', interarrival_rate_multiplier=interarrival_rate_multiplier)  # wrapped_env(running_time=running_time)
    return env

def make_hetero_undirected(hetero_data):
    for edge_type in hetero_data.edge_types:
        source_type, rel_type, target_type = edge_type
        edge_index = hetero_data[source_type, rel_type, target_type].edge_index
        edge_attr = hetero_data[source_type, rel_type, target_type].edge_attr
        reverse_edge_index = edge_index[[1, 0]]
        hetero_data[target_type, rel_type, source_type].edge_index = reverse_edge_index
        hetero_data[target_type, rel_type, source_type].edge_attr = edge_attr #this stays the same
    return hetero_data


def make_hetero_undirected_optimized(hetero_data):
    for edge_type in hetero_data.edge_types:
        source_type, rel_type, target_type = edge_type
        edge_index = hetero_data[source_type, rel_type, target_type].edge_index
        edge_attr = hetero_data[source_type, rel_type, target_type].edge_attr
        reverse_edge_index = edge_index[[1, 0]]

        # Concatenate the original and reversed edge indices and attributes
        hetero_data[source_type, rel_type, target_type].edge_index = torch.cat((edge_index, reverse_edge_index), dim=1)
        hetero_data[source_type, rel_type, target_type].edge_attr = torch.cat((edge_attr, edge_attr), dim=0)
    return hetero_data


def add_self_loops_to_assignment_nodes(data):
    sequence = torch.arange(data['assignment'].x.size(0), dtype=torch.int)
    assignment_edge_index_with_self_loops = torch.stack([sequence, sequence])
    data['assignment', 'edge', 'assignment'].edge_index = assignment_edge_index_with_self_loops
    return data


def batch_to_vectors(batch):
    processed = []
    masks = []

    for el in batch:
        # Resource status from 'resources' tensor
        resource_status = el['resources'][:, 0]  # First column is availability

        # Resource-activity mapping
        n_resources = el['resources'].shape[0]
        n_activities = el['task_types'].shape[0]
        resource_activity = torch.zeros(n_resources)
        edge_index = el['reconstruct_edges']

        for i in range(edge_index.shape[1]):
            if el['mask'][i]:
                resource_idx = edge_index[0, i]
                activity_idx = edge_index[1, i]
                resource_activity[resource_idx] = activity_idx / n_activities

        # Queue sizes from 'task_types', normalized
        queue_sizes = el['task_types']
        queue_sizes = torch.minimum(queue_sizes * 10.0, torch.ones_like(queue_sizes))

        # Combine features
        features = torch.cat([
            resource_status,
            resource_activity,
            queue_sizes
        ])
        processed.append(features)

        # Create mask for valid actions
        masks.append(el['mask'])

    # Create Tianshou batch
    return ts.data.Batch(
        obs=torch.stack(processed),
        mask=torch.stack(masks)
    )


class VectorObservationNet(nn.Module):
    def __init__(self, num_resources, num_activities):
        super().__init__()
        self.input_dim = num_resources + num_resources + num_activities
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

    def forward(self, x):
        if isinstance(x, ts.data.Batch):
            x = batch_to_vectors(x)
        elif isinstance(x, dict) or isinstance(x, list):
            x = batch_to_vectors(x)
        return self.network(x.obs), x.mask

class VectorActorNet(nn.Module):
    def __init__(self, num_resources, num_activities, num_edges):
        super().__init__()
        self.base = VectorObservationNet(num_resources, num_activities)
        self.num_edges = num_edges #+ 1  # +1 for no-op action
        self.policy_head = nn.Linear(64, self.num_edges)

    def forward(self, x, state=None, info={}):
        features, mask = self.base(x)
        logits = self.policy_head(features)

        # Use the mask from preprocessing if available
        logits = logits.masked_fill(~mask, float('-inf'))

        probs = F.softmax(logits, dim=-1)
        return probs, state

class VectorCriticNet(nn.Module):
    def __init__(self, num_resources, num_activities):
        super().__init__()
        self.base = VectorObservationNet(num_resources, num_activities)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x, state=None, info={}):
        features, _ = self.base(x)
        value = self.value_head(features)
        return value



if __name__ == '__main__':
    # if true, load model for a new round of training

    if problem == 'toloka':
        n_edges = 132
    elif problem == 'bpi2017':
        n_edges = 573
        n_activities = 7
        n_resources = 145
    elif problem == 'bpi2012':
        n_edges = 199
    elif problem == 'bpi2018':
        n_edges = 479  # full dataset: 479, with threshold==3000: 181
    elif problem == 'consulta':
        n_edges = 435
    elif problem == 'production':
        n_edges = 76
    elif problem == 'microsoft':
        n_edges = 55
        n_activities = 13
        n_resources = 8
    else:
        raise Exception("Invalid problem name")



    # Create log dir
    log_dir = f"./tmp/{problem}"
    os.makedirs(log_dir, exist_ok=True)

    # Initialize networks and policy
    actor_net = VectorActorNet(num_resources=n_resources, num_activities=n_activities, num_edges=n_edges) #+ 1)
    critic_net = VectorCriticNet(num_resources=n_resources, num_activities=n_activities)

    optim = torch.optim.Adam(
        list(actor_net.parameters()) + list(critic_net.parameters()),
        lr=train_args["lr"]
    )

    policy = PPOPolicy(
        actor_net, critic_net, optim,
        discount_factor=train_args["discount_factor"],
        dist_fn=torch.distributions.categorical.Categorical,
        deterministic_eval=True,
        reward_normalization=False
    )

    scheduler = ExponentialLR(optim, 0.95)

    policy = PPOPolicy(actor_net, critic_net, optim,
                       discount_factor=train_args["discount_factor"],
                       dist_fn=torch.distributions.categorical.Categorical,
                       deterministic_eval=True,
                       #lr_scheduler=scheduler,
                       reward_normalization=False
                       )
    policy.action_type = "discrete"
    log_path = os.path.join(f"logs/ppo_vector/{problem}")

    #create folder if necessary
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    train_envs = ts.env.DummyVectorEnv(
        [lambda: get_env(problem, running_time, problem_type) for _ in range(train_args["nr_envs"])]
    )

    collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(train_args['buffer_size'], train_args["nr_envs"]),
                                  exploration_noise=False, preprocess_fn=preprocess_function)
    collector.reset()

    test_envs = ts.env.DummyVectorEnv(
        [lambda: get_env(problem, running_time, problem_type) for _ in range(train_args["nr_envs"])]
    )

    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=False, preprocess_fn=preprocess_function)
    test_collector.reset()

    if load_model:
        policy.load_state_dict(torch.load(f"ppo_graph_{problem}.pt"))


    print("Starting training")
    policy.train()
    if problem == 'bpi2017':
        trainer = ts.trainer.OnpolicyTrainer(
            policy, collector, test_collector=test_collector,
            max_epoch=train_args["max_epoch"],
            step_per_epoch=train_args["step_per_epoch"],
            #step_per_collect=train_args["step_per_collect"],
            episode_per_collect=train_args["episode_per_collect"],
            episode_per_test=20, batch_size=train_args["batch_size"],
            repeat_per_collect=train_args["repeat_per_collect"],
            logger=logger, test_in_train=True, verbose=False,
            save_best_fn=save_best_fn)
    else:
        trainer = ts.trainer.OnpolicyTrainer(
            policy, collector, test_collector=test_collector,
            max_epoch=train_args["max_epoch"],
            step_per_epoch=train_args["step_per_epoch"],
            step_per_collect=train_args["step_per_collect"],
            #episode_per_collect=train_args["episode_per_collect"],
            episode_per_test=100, batch_size=train_args["batch_size"],
            repeat_per_collect=train_args["repeat_per_collect"],
            logger=logger, test_in_train=True, verbose=False,
            save_best_fn=save_best_fn)

    result = trainer.run()
    print(f'Finished training!')