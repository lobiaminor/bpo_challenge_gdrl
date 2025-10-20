import numpy as np
import torch
from tianshou.policy import PPOPolicy

from simulator import Simulator
from planners import RandomPlanner, ShortestProcessingTime, FIFOProcess, FIFOActivity, ShortestProcessingTimeStandardized
from planners import PPOPlannerTianshou as PPOPlanner
from time import time

from train_graph import train_args, HeteroActorNodeSelection, HeteroCriticNodeSelection
from train_vector import train_args, VectorCriticNet, VectorActorNet


problem_name = "microsoft" #possible values bpi2012, bpi2017, consulta, production, microsoft
problem_type = 'regenerated'  # possible values original, regenerated

# possible values: ppo, spt, fifo, bayes, random
multiple_models = ['ppo_vector']#['random', 'fifo_process', 'spt', 'ppo_vector', 'ppo']
num_replicates = 100
running_time = 7 * 24


if problem_name == 'bpi2017':
    n_edges = 573
    n_activities = 7
    n_resources = 145
    if problem_type == 'original':
        instance_file = "./BPI Challenge 2017 - instance.pickle"
    else:
        instance_file = "./data/bpi2017_problem.pkl"
elif problem_name == 'bpi2012':
    instance_file = "./data/bpi2012_problem.pkl"
    n_edges = 199
    n_activities = 6
    n_resources = 52
elif problem_name == 'consulta':
    instance_file = "./data/consulta.pkl"
    n_edges = 435
    n_activities = 16
    n_resources = 179
elif problem_name == 'production':
    instance_file = "data/production.pkl"
    n_edges = 76
    n_activities = 13
    n_resources = 33
elif problem_name == 'microsoft':
    instance_file = "data/microsoft.pkl"
    n_edges = 55
    n_activities = 13
    n_resources = 8
else:
    raise Exception("Invalid problem name")


def preprocess_function(normalize=True, **kwargs):
    def normalize_features(x):
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        return (x - mean) / (std + 1e-6)

    if normalize:
        kwargs["graph_dict"] = {
            "resources": normalize_features(torch.from_numpy(kwargs["graph_dict"]["resources"]).float()),
            "task_types": normalize_features(torch.from_numpy(kwargs["graph_dict"]["task_types"]).float()),
            "assignments": normalize_features(torch.from_numpy(kwargs["graph_dict"]["assignments"]).float()),
            "reconstruct_edges": torch.from_numpy(kwargs["graph_dict"]["reconstruct_edges"]).long(),
            "mask": torch.from_numpy(kwargs["graph_dict"]["mask"]).bool()
        }
    else:
        kwargs["graph_dict"] = {
            "resources": torch.from_numpy(kwargs["graph_dict"]["resources"]).float(),
            "task_types": torch.from_numpy(kwargs["graph_dict"]["task_types"]).float(),
            "assignments": torch.from_numpy(kwargs["graph_dict"]["assignments"]).float(),
            "reconstruct_edges": torch.from_numpy(kwargs["graph_dict"]["reconstruct_edges"]).long(),
            "mask": torch.from_numpy(kwargs["graph_dict"]["mask"]).bool()
        }
    return kwargs


def simulate_competition():
    for model_name in multiple_models:
        if model_name == 'ppo':

            actor_net = HeteroActorNodeSelection(n_edges=n_edges)
            critic_net = HeteroCriticNodeSelection(n_edges=n_edges)
            optim = torch.optim.Adam(list(actor_net.parameters()) + list(critic_net.parameters()), lr=1e-4)

            policy = PPOPolicy(actor_net, critic_net, optim,
                               discount_factor=train_args["discount_factor"],
                               max_batchsize=256,  # max batch size for GAE estimation, default to 256
                               value_clip=True,
                               dist_fn=torch.distributions.categorical.Categorical,
                               deterministic_eval=True,
                               # lr_scheduler=scheduler,
                               reward_normalization=False
                               )

            my_planner = PPOPlanner(policy,
                                    preprocess_fn=preprocess_function,
                                    actor_network_name=f"ppo_graph_{problem_name}")
        elif model_name == 'ppo_vector':
            actor_net = VectorActorNet(num_resources=n_resources, num_activities=n_activities, num_edges=n_edges, allow_postpone=True)
            critic_net = VectorCriticNet(num_resources=n_resources, num_activities=n_activities, allow_postpone=True)
            optim = torch.optim.Adam(list(actor_net.parameters()) + list(critic_net.parameters()), lr=1e-4)

            policy = PPOPolicy(actor_net, critic_net, optim,
                               discount_factor=train_args["discount_factor"],
                               max_batchsize=256,  # max batch size for GAE estimation, default to 256
                               value_clip=True,
                               dist_fn=torch.distributions.categorical.Categorical,
                               deterministic_eval=True,
                               # lr_scheduler=scheduler,
                               reward_normalization=False
                               )

            my_planner = PPOPlanner(policy,
                                    preprocess_fn=preprocess_function,
                                    actor_network_name=f"ppo_vector_{problem_name}")
        elif model_name == 'spt':
            my_planner = ShortestProcessingTime()
        elif model_name == 'spt_std':
            my_planner = ShortestProcessingTimeStandardized()
        elif model_name == 'fifo_process':
            my_planner = FIFOProcess()
        elif model_name == 'fifo_activity':
            my_planner = FIFOActivity()
        elif model_name == 'random':
            my_planner = RandomPlanner()
        else:
            raise Exception("Invalid model_name")

        results = []
        times = []
        for j in range(num_replicates):
            print(j)
            simulator = Simulator(running_time=running_time, planner=my_planner,
                                  instance_file=instance_file, deterministic_processing=False)

            if type(my_planner) == PPOPlanner or type(my_planner) == ShortestProcessingTime or type(my_planner) == ShortestProcessingTimeStandardized:
                my_planner.linkSimulator(simulator)

            t1 = time()
            result = simulator.run()
            # print(f'Simulation finished in {time()-t1} seconds')
            print(f"Completed tasks: {simulator.total_completed_tasks}")
            times.append(time() - t1)
            results.append(result)
            print(f'Running average: {np.mean(results)}')

        print(f"Results for model: {model_name}")
        print(f"Average and std cycle time: {np.mean(results)} +- {np.std(results)}")
        #Calculate 95% confidence interval
        print(f"Average and 95% confidence interval: {np.mean(results)} +- {1.96*np.std(results)/np.sqrt(num_replicates)}")
        print(f"Average and 99% confidence interval: {np.mean(results)} +- {2.58*np.std(results)/np.sqrt(num_replicates)}")

        print(f"Average time: {np.mean(times)} +- {np.std(times)}")

        #Write the same things to a file
        with open(f"results_{problem_name}_{running_time}.txt", "a") as f:
            f.write(f"Results for model: {model_name}\n")
            f.write(f"Average and std cycle time: {np.mean(results)} +- {np.std(results)}\n")
            f.write(f"Average and 95% confidence interval: {np.mean(results)} +- {1.96*np.std(results)/np.sqrt(num_replicates)}\n")
            f.write(f"Average and 99% confidence interval: {np.mean(results)} +- {2.58*np.std(results)/np.sqrt(num_replicates)}\n")
            f.write(f"Average time: {np.mean(times)} +- {np.std(times)}\n")


if __name__ == "__main__":
    simulate_competition()
