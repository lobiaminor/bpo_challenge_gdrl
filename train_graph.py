import os
import torch

import numpy as np
import tianshou as ts

from torch import nn, softmax
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import HANConv
from torch_geometric.data import HeteroData, Batch
from torch_geometric.utils import scatter
from tianshou.utils import TensorboardLogger
from tianshou.policy import PPOPolicy
from torch.optim.lr_scheduler import ExponentialLR, LinearLR

from bpo_env_graph import BPOEnv

# possible problems are: fines, bpi2017, bpi2018
problems = ['consulta']
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
              "nr_envs": 1,  # 64
              "max_epoch": 100,
              "buffer_size": 200000,
              "step_per_epoch": 20000,
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


def batch_to_hetero(batch, make_undirected=False, add_self_loops=True, remove_mask = True, reduce_graph=True):
    graph_list = []
    mask = []

    for el in batch:
        graph = HeteroData()
        graph['resource'].x = el['resources']
        graph['task_type'].x = el['task_types'].unsqueeze_(1)
        graph['assignment'].x = el['assignments']

        if not remove_mask:
            graph['resource', 'resource_task_type', 'task_type'].edge_index = el['edge_index']
            graph['resource', 'resource_task_type', 'task_type'].edge_attr = el['edge_attr']
        else: # remove mask
            #keeping the whole graph
            if not reduce_graph:
                graph['resource', 'resource_task_type', 'task_type'].edge_index = el['edge_index']
                graph['resource', 'resource_task_type', 'task_type'].edge_attr = el['edge_attr'][:, -1]
                mask = el['edge_attr'][:, 0]
            else: #drop the edges that are not in the mask
                temp_mask = el['mask'].float()
                mask.append(temp_mask)
                # Add 'assignment' nodes with edges incoming from resources and task types corresponding to the assignment
                selected_edges = torch.unbind(el['reconstruct_edges'][el['mask']], dim=1)

                graph['resource', 'edge', 'assignment'].edge_index = torch.stack([selected_edges[0], torch.where(temp_mask)[0]])
                graph['task_type', 'edge', 'assignment'].edge_index = torch.stack([selected_edges[1], torch.where(temp_mask)[0]])

        if make_undirected:
            graph = make_hetero_undirected(graph)

        if add_self_loops:
            graph = add_self_loops_to_assignment_nodes(graph)
        graph_list.append(graph)

    return graph_list, torch.stack(mask)


def batch_to_hetero_postpone(batch, make_undirected=False, add_self_loops=True, remove_mask=True, reduce_graph=True):
    graph_list = []
    mask = []

    for el in batch:
        graph = HeteroData()
        graph['resource'].x = el['resources']
        graph['task_type'].x = el['task_types'].unsqueeze_(1)

        # Add no-op node with a feature vector of zeros
        noop_feature = torch.zeros(1, el['assignments'].size(1), device=el['assignments'].device)
        assignments_with_noop = torch.cat([el['assignments'], noop_feature], dim=0)
        graph['assignment'].x = assignments_with_noop

        if not remove_mask:
            graph['resource', 'resource_task_type', 'task_type'].edge_index = el['edge_index']
            graph['resource', 'resource_task_type', 'task_type'].edge_attr = el['edge_attr']
        else:
            if not reduce_graph:
                graph['resource', 'resource_task_type', 'task_type'].edge_index = el['edge_index']
                graph['resource', 'resource_task_type', 'task_type'].edge_attr = el['edge_attr'][:, -1]
                mask = el['edge_attr'][:, 0]
            else:
                temp_mask = el['mask'].float()
                # Add 1 to allow selecting no-op
                noop_mask = torch.ones(1, device=temp_mask.device)
                temp_mask_with_noop = torch.cat([temp_mask, noop_mask])
                mask.append(temp_mask_with_noop)

                selected_edges = torch.unbind(el['reconstruct_edges'][el['mask']], dim=1)

                # Add edges for the no-op node
                noop_idx = assignments_with_noop.size(0) - 1
                num_resources = el['resources'].size(0)
                num_task_types = el['task_types'].size(0)

                # Connect all resources to no-op node
                resource_noop_edges = torch.stack([
                    torch.arange(num_resources, device=el['resources'].device),
                    torch.full((num_resources,), noop_idx, device=el['resources'].device)
                ])

                # Connect all task types to no-op node
                task_type_noop_edges = torch.stack([
                    torch.arange(num_task_types, device=el['task_types'].device),
                    torch.full((num_task_types,), noop_idx, device=el['task_types'].device)
                ])

                # Combine original edges with no-op edges
                resource_edges = torch.cat([
                    torch.stack([selected_edges[0], torch.where(temp_mask)[0]]),
                    resource_noop_edges
                ], dim=1)

                task_type_edges = torch.cat([
                    torch.stack([selected_edges[1], torch.where(temp_mask)[0]]),
                    task_type_noop_edges
                ], dim=1)

                graph['resource', 'edge', 'assignment'].edge_index = resource_edges
                graph['task_type', 'edge', 'assignment'].edge_index = task_type_edges

        if make_undirected:
            graph = make_hetero_undirected(graph)

        if add_self_loops:
            graph = add_self_loops_to_assignment_nodes(graph)
        graph_list.append(graph)

    return graph_list, torch.stack(mask)


class HeteroActorNodeSelection(torch.nn.Module):
    def __init__(self, hidden_dim=1, n_edges=573, metadata=(['task_type', 'resource', 'assignment'], [('resource', 'edge', 'assignment'), ('task_type', 'edge', 'assignment'), ('assignment', 'edge', 'assignment')]), n_graph_layers=1, n_edge_layers=1, input_dim=1, output_size=1):
        super().__init__()

        self.num_edges = n_edges

        #encoder
        self.conv = HANConv({'task_type': 1, 'resource': 2, 'assignment': 1}, hidden_dim, heads=1, metadata=metadata)

        #decoder
        self.lin_actor1 = nn.Linear(hidden_dim*2, 64)
        self.batch_norm_actor = nn.BatchNorm1d(64)
        self.relu_actor = nn.ReLU()
        self.lin_actor2 = nn.Linear(64, 1)

    def forward(self, data, state=None, info={}):
        graph_list, mask = batch_to_hetero(data)
        batch = Batch.from_data_list(graph_list)

        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, 0)

        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict

        x_dict = self.conv(x_dict, edge_index_dict)
        x = x_dict['assignment'] #purely attentive (no aggregation)

        if not self.training:
            mask = mask.squeeze(0)
            mask = mask.unsqueeze(-1)
            x.add_(mask)
            x = x.reshape(-1, self.num_edges)
            x = softmax(x, dim=1)
        else:
            x = x.reshape(-1, self.num_edges)
            x.add_(mask)
            x = softmax(x, dim=1)

        return x, state


class HeteroCriticNodeSelection(torch.nn.Module):
    def __init__(self, hidden_dim=16, n_edges=573, metadata=(['task_type', 'resource', 'assignment'], [('resource', 'edge', 'assignment'), ('task_type', 'edge', 'assignment'), ('assignment', 'edge', 'assignment')])):
        super().__init__()

        self.n_edges = n_edges

        self.conv = HANConv({'task_type': 1, 'resource': 2, 'assignment': 1}, hidden_dim, heads=1, metadata=metadata)

        self.lin = nn.Linear(16, 1)

        #unused
        self.lin_critic1 = nn.Linear(32, 64)
        self.batch_norm_critic = nn.BatchNorm1d(64)
        self.relu_critic = nn.ReLU()
        self.lin_critic2 = nn.Linear(64, 1)

    def forward(self, data, state=None, info={}):
        graph_list, mask = batch_to_hetero(data)
        batch = Batch.from_data_list(graph_list)

        index = batch['assignment']['batch']
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict

        x_dict = self.conv(x_dict, edge_index_dict)
        x = x_dict['assignment'] #purely attentive (e.g. no aggregation)

        if not self.training:
            x = sum(x)
        else:

            x = scatter(x, index, dim=0, reduce='sum')

        x = self.lin(x)

        return x

if __name__ == '__main__':


    for problem in problems:

        if problem == 'toloka':
            n_edges = 132
        elif problem == 'bpi2017':
            n_edges = 573
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
        else:
            raise Exception("Invalid problem name")



        # Create log dir
        log_dir = f"./tmp/{problem}"
        os.makedirs(log_dir, exist_ok=True)

        actor_net = HeteroActorNodeSelection(n_edges=n_edges)
        critic_net = HeteroCriticNodeSelection(n_edges=n_edges)

        optim = torch.optim.Adam(
            params=list(actor_net.parameters()) + list(critic_net.parameters()),
            lr=train_args["lr"]
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
        log_path = os.path.join(f"logs/ppo_graph/{problem}")

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