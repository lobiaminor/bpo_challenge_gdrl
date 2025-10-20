#this script is used to generate traces for the different problems (if generate_trace is set to True)
#or to check the differences between the actions taken by the PPO and SPT policies on the traces (if generate_trace is set to False)

import json

import numpy as np
import torch
from tianshou.policy import PPOPolicy

from simulator import Simulator
from planners import RandomPlanner, ShortestProcessingTime, FIFOProcess, FIFOActivity, ShortestProcessingTimeStandardized
from planners import PPOPlannerTianshou as PPOPlanner
import matplotlib.pyplot as plt
import numpy as np

from train_graph import train_args, HeteroActorNodeSelection, HeteroCriticNodeSelection

import seaborn as sns

# Set Seaborn style and context
sns.set_style("whitegrid")
sns.set_context("talk")

#generate a set of traces (or False to compute and plot the differences between spt and ppo for every problem)
generate_trace = False


# possible values: ppo, spt, fifo, bayes, random
running_time = 7 * 24
replicates = 10
#max num samples to get
num_samples = 1000 #0 means no limit (the whole set of collected states)


problem_type = 'regenerated'  # possible values original, regenerated
multiple_models = ['ppo', 'spt']


#constants
all_problems = ['bpi2012', 'bpi2017', 'microsoft', 'consulta', 'production']


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
            "mask": torch.from_numpy(np.array(kwargs["graph_dict"]["mask"])).bool()
        }
    else:
        kwargs["graph_dict"] = {
            "resources": torch.from_numpy(kwargs["graph_dict"]["resources"]).float(),
            "task_types": torch.from_numpy(kwargs["graph_dict"]["task_types"]).float(),
            "assignments": torch.from_numpy(kwargs["graph_dict"]["assignments"]).float(),
            "reconstruct_edges": torch.from_numpy(kwargs["graph_dict"]["reconstruct_edges"]).long(),
            "mask": torch.from_numpy(np.array(kwargs["graph_dict"]["mask"])).bool()
        }
    return kwargs


def get_data(problem_name):
    if problem_name == 'toloka':
        instance_file = "./data/toloka_problem.pkl"
        n_edges = 132
    elif problem_name == 'fines':
        instance_file = "./data/fines_problem.pkl"
        n_edges = 53
    elif problem_name == 'bpi2017':
        n_edges = 573
        if problem_type == 'original':
            instance_file = "./BPI Challenge 2017 - instance.pickle"
        else:
            instance_file = "./data/bpi2017_problem.pkl"
    elif problem_name == 'bpi2012':
        instance_file = "./data/bpi2012_problem.pkl"
        n_edges = 199
    elif problem_name == 'bpi2018':
        instance_file = "./data/bpi2018_problem.pkl"
        n_edges = 479
    elif problem_name == 'consulta':
        instance_file = "./data/consulta.pkl"
        n_edges = 435
    elif problem_name == 'production':
        instance_file = "data/production.pkl"
        n_edges = 76
    elif problem_name == 'microsoft':
        instance_file = "data/microsoft.pkl"
        n_edges = 55
    else:
        raise Exception(f"Invalid problem name {problem_name}")

    return instance_file, n_edges

def generate_traces():

    for problem_name in all_problems:
        for i in range(replicates):
            instance_file, n_edges = get_data(problem_name)

            simulator = Simulator(running_time=running_time, planner=RandomPlanner(),
                                  instance_file=instance_file, record_states=True, max_transitions=num_samples)
            simulator.run()


def check_differences():
    values_to_plot = {}
    for problem_name in all_problems:
        instance_file, n_edges = get_data(problem_name)

        actor_net = HeteroActorNodeSelection(n_edges=n_edges)
        critic_net = HeteroCriticNodeSelection(n_edges=n_edges)
        optim = torch.optim.Adam(list(actor_net.parameters()) + list(critic_net.parameters()), lr=1e-4)

        policy = PPOPolicy(actor_net, critic_net, optim,
                           discount_factor=train_args["discount_factor"],
                           max_batchsize=64,  # max batch size for GAE estimation, default to 256
                           value_clip=True,
                           dist_fn=torch.distributions.categorical.Categorical,
                           deterministic_eval=True,
                           # lr_scheduler=scheduler,
                           reward_normalization=False
                           )

        ppo_planner = PPOPlanner(policy,
                                 preprocess_fn=preprocess_function,
                                 actor_network_name=f"ppo_graph_{problem_name}")

        spt_planner = ShortestProcessingTime()

        simulator = Simulator(running_time=1, planner=None,
                              instance_file=instance_file)

        ppo_planner.linkSimulator(simulator)
        spt_planner.linkSimulator(simulator)

        original_problem_name = problem_name
        if problem_name in ['bpi2012', 'bpi2017', 'bpi2018']:
            problem_name = f'{problem_name}_problem'

        with open(f'states_{problem_name}.json', 'r') as f:
            # record when the difference is detected
            n_differences = 0
            n_equal = 0

            #total number of observations considered
            n_obs = 0

            #for each entry in the dataset, try both planners
            for line in f:

                entry = json.loads(line)
                ppo_act = ppo_planner.plan_from_trace(entry['available_resources'], entry['unassigned_tasks'], entry['busy_resources'], entry['mask'])
                entry = json.loads(line)
                spt_act = spt_planner.plan_from_trace(entry['available_resources'], entry['unassigned_tasks'], simulator.problem_resource_pool)

                if ppo_act != spt_act:
                    n_differences += 1
                else:
                    n_equal += 1

                n_obs += 1
                if num_samples > 0 and n_obs >= num_samples:
                    break

        print(f'Tot number of obs {n_obs}')
        #compute percentage
        n_differences = n_differences / n_obs
        n_equal = n_equal / n_obs

        #save to plot all together
        values_to_plot[original_problem_name] = [n_differences, n_equal]
        print(f"Problem {original_problem_name} has {n_differences} differences and {n_equal} equal actions")

    problem_dict = {'bpi2012': "BPI2012", 'bpi2017': "BPI2017", 'consulta': "CONS", 'production': "PROD",
                    "microsoft": "MICRO"}
    x = np.arange(len(all_problems))
    #plot one bar for each problem, with the number of differences and the number of equal actions in red and blue
    #use problem_name for the x axis

    weight_counts = {
        "Equal": np.array([values_to_plot[problem][1] for problem in all_problems]),
        "Different": np.array([values_to_plot[problem][0] for problem in all_problems]),
    }
    width = 0.5

    fig, ax = plt.subplots()
    bottom = np.zeros(len(all_problems))

    for boolean, weight_count in weight_counts.items():
        p = ax.bar([problem_dict[prob] for prob in all_problems], weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count

    #ax.set_title("Comparison of actions between PPO and SPT")

    #move legend to the right outside the graph
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

    plt.tight_layout()
    plt.savefig('differences.pdf')
    plt.show()


def plot_saved_results():
    with open('results.json', 'r') as f:
        results = json.load(f)

    problem_dict = {'bpi2012': "BPI2012", 'bpi2017': "BPI2017", 'consulta': "CONS", 'production': "PROD", "microsoft": "MICRO"}

    #results = {'bpi2012': [0.0, 1.0], 'bpi2017': [0.2864391694824484, 0.7135608305175516], 'bpi2018': [0.10443282174397668, 0.8955671782560233], 'consulta': [0.010851470497564585, 0.9891485295024354], 'production': [0.04435815591167857, 0.9556418440883214]}

    #plot one bar for each problem, with the number of differences and the number of equal actions in red and blue
    #use problem_name for the x axis

    weight_counts = {
        "Equal": np.array([results[problem][1] for problem in all_problems]),
        "Different": np.array([results[problem][0] for problem in all_problems]),
    }
    width = 0.5

    fig, ax = plt.subplots()
    bottom = np.zeros(len(all_problems))

    for boolean, weight_count in weight_counts.items():
        p = ax.bar([problem_dict[prob] for prob in all_problems], weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count

    #ax.set_title("Comparison of actions between PPO and SPT")
    #legend should appear outside the graph on top of the image
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)


    #Save image
    # save to file in vector format
    plt.tight_layout()
    plt.savefig('differences.pdf')
    plt.show()





if __name__ == "__main__":
    if generate_trace:
        generate_traces()
    else:
        check_differences()
        #plot_saved_results()

