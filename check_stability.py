#this script runs the simulation for the different problems under the spt policy
#the objective is to plot the amount of cases in the system an any point in time for each problem
#the idea is to check if the system reaches stability

import numpy as np
import torch
from matplotlib import pyplot as plt
from tianshou.policy import PPOPolicy

from simulator import Simulator
from planners import RandomPlanner, ShortestProcessingTime, FIFOProcess, FIFOActivity, ShortestProcessingTimeStandardized
from planners import PPOPlannerTianshou as PPOPlanner
from time import time

from train_graph import train_args, HeteroActorNodeSelection, HeteroCriticNodeSelection

import seaborn as sns

# Set Seaborn style and context
sns.set_style("whitegrid")
sns.set_context("talk")

#load data from previous runs
load_data = True

# if load_data==False, specify which problem to generate the trace for
problems = ['bpi2012', 'bpi2017', 'microsoft', 'consulta', 'production']
problem_type = 'regenerated'  # possible values original, regenerated


multiple_models = ['spt']
num_replicates = 100
running_time = 24 * 7 * 4





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


def compute_average(values, indexes, expected_length=running_time+1):
    index_sum_count = {}

    for value, index in zip(values, indexes):
        if index not in index_sum_count:
            index_sum_count[index] = {'sum': 0, 'count': 0}
        index_sum_count[index]['sum'] += value
        index_sum_count[index]['count'] += 1

    index_avg = {index: index_sum_count[index]['sum'] / index_sum_count[index]['count'] for index in index_sum_count}

    #fill the missing indexes with 0
    for i in range(expected_length):
        if i not in index_avg:
            index_avg[i] = 0

    #remove extra indexes
    index_avg = {k: v for k, v in index_avg.items() if k < expected_length}

    #if len(index_avg) != expected_length:
    #    raise Exception(f'Expected length {expected_length}, got {len(index_avg )}')

    return list(index_avg.values())
def plot_cases(total_cases_dict):

    for el in total_cases_dict:
        #convert the first element of each entry to integer
        el['time'] = [int(i) for i in el['time']]
        #compute the average of the second element of each entry using the first element as the key
        average_indexed = compute_average(el['total_cases'], el['time'])
        #replace the second element of each entry with the average
        el['total_cases'] = average_indexed

    #x axys is the average time, considering a step of 1 second
    x_ax = np.arange(0, running_time+1, 1)

    #y axys is the average total cases in the system considering a step of 1 hour
    y_ax = np.mean([el['total_cases'] for el in total_cases_dict], axis=0)


    # Plot the number of cases in the system over time
    plt.plot(x_ax, y_ax)
    plt.xlabel('Time')
    plt.ylabel('Number of cases in the system')
    plt.title('Number of cases in the system over time')
    plt.show()

    #save x_ax and y_ax to a file
    np.save(f'./data/{problem_name}_cases.npy', y_ax)
    np.save(f'./data/{problem_name}_time.npy', x_ax)

def simulate_competition(problem_name):
    for model_name in multiple_models:
        if model_name == 'ppo':

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

            my_planner = PPOPlanner(policy,
                                    preprocess_fn=preprocess_function,
                                    actor_network_name=f"ppo_graph_{problem_name}")
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


        print(f"Generating convergence plot for {problem_name}")
        results = []
        times = []

        average_cases = []
        for j in range(num_replicates):
            print(j)
            simulator = Simulator(running_time=running_time, planner=my_planner,
                                  instance_file=instance_file, record_total_cases=True)

            if type(my_planner) == PPOPlanner or type(my_planner) == ShortestProcessingTime or type(my_planner) == ShortestProcessingTimeStandardized:
                my_planner.linkSimulator(simulator)

            t1 = time()
            result = simulator.run()
            # print(f'Simulation finished in {time()-t1} seconds')
            print(f"Completed tasks: {simulator.total_completed_tasks}")
            times.append(time() - t1)
            results.append(result)
            print(f'Running average: {np.mean(results)}')

            average_cases.append(simulator.total_cases_dict)

        plot_cases(average_cases)


def plot_all_graphs_from_npy():
    #in one image we should have five graphs, one per problem

    problem_dict = {'bpi2012': "BPI2012", 'bpi2017': "BPI2017", 'consulta': "CONS", 'production': "PROD", 'microsoft': "MICRO"}

    # Increase the figure size to accommodate the legend
    plt.figure(figsize=(12, 6))

    for problem in problems: #'bpi2012', 'bpi2017', 'consulta', 'production', 'toloka'
        y_ax = np.load(f'./data/{problem}_cases.npy')
        x_ax = np.load(f'./data/{problem}_time.npy')

        # Create a single image with the number of cases in the system over time for all problems
        plt.plot(x_ax, y_ax, label=problem_dict[problem])


    plt.xlabel('Time')
    plt.ylabel('Number of cases')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim(0, max(x_ax))
    plt.tight_layout()
    plt.savefig('./data/all_problems_cases.pdf')
    plt.show()



if __name__ == "__main__":
    if not load_data:
        for problem_name in problems:
            if problem_name == 'toloka':
                instance_file = "./data/toloka_problem.pkl"
                n_edges = 62
            elif problem_name == 'fines':
                instance_file = "./data/fines_problem.pkl"
                n_edges = 53
            elif problem_name == 'bpi2017':
                n_edges = 2057#573
                if problem_type == 'original':
                    instance_file = "./BPI Challenge 2017 - instance.pickle"
                else:
                    instance_file = "./data/bpi2017_problem.pkl"
            elif problem_name == 'bpi2012':
                instance_file = "./data/bpi2012_problem.pkl"
                #n_edges = 199
                n_edges = 219
            elif problem_name == 'bpi2018':
                instance_file = "./data/bpi2018_problem.pkl"
                n_edges = 479
            elif problem_name == 'consulta':
                instance_file = "./data/consulta.pkl"
                n_edges = 435
                # n_edges = 439
            elif problem_name == 'production':
                instance_file = "data/production.pkl"
                n_edges = 78
            elif problem_name == 'microsoft':
                instance_file = "data/microsoft.pkl"
                n_edges = 78
            else:
                raise Exception("Invalid problem name")

            simulate_competition(problem_name)
    else:
        plot_all_graphs_from_npy()