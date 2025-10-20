import json
import torch
import copy
import random

import numpy as np
import pandas as pd

from itertools import permutations, combinations
from simulator import Simulator
from abc import ABC, abstractmethod


class Planner(ABC):
    """Abstract class that all planners must implement."""

    @abstractmethod
    def plan(self, available_resources, unassigned_tasks, resource_pool):
        """
        Assign tasks to resources from the simulation environment.

        :param environment: a :class:`.Simulator`
        :return: [(task, resource, moment)], where
            task is an instance of :class:`.Task`,
            resource is one of :attr:`.Problem.resources`, and
            moment is a number representing the moment in simulation time
            at which the resource must be assigned to the task (typically, this can also be :attr:`.Simulator.now`).
        """
        raise NotImplementedError


# Greedy assignment
class GreedyPlanner(Planner):
    """A :class:`.Planner` that assigns tasks to resources in an anything-goes manner."""

    def plan(self, available_resources, unassigned_tasks, resource_pool):
        assignments = []
        available_resources = available_resources.copy()
        # assign the first unassigned task to the first available resource, the second task to the second resource, etc.
        for task in unassigned_tasks:
            for resource in available_resources:
                if resource in resource_pool[task.task_type]:
                    available_resources.remove(resource)
                    assignments.append((task, resource))
                    break
        return assignments

    def report(self, event):
        pass  # print(event)


class RandomPlanner(Planner):
    """A :class:`.Planner` that assigns tasks to resources in an anything-goes manner."""

    def get_possible_assignments(self, available_resources, unassigned_tasks, resource_pool):
        possible_assignments = []
        for task in unassigned_tasks:
            for resource in available_resources:
                if resource in resource_pool[task.task_type]:
                    possible_assignments.append((resource, task))
        return possible_assignments

    def plan(self, available_resources, unassigned_tasks, resource_pool):
        available_resources = available_resources.copy()
        unassigned_tasks = unassigned_tasks.copy()
        assignments = []

        possible_assignments = self.get_possible_assignments(available_resources, unassigned_tasks, resource_pool)

        # while len(possible_assignments) > 0:
        assignment = random.choice(possible_assignments)
        unassigned_tasks.remove(assignment[1])
        available_resources.remove(assignment[0])
        assignment = (assignment[0], assignment[1].task_type)
        assignments.append(assignment)

        return assignments

    def report(self, event):
        pass  # print(event)


class ShortestProcessingTimeStandardized(Planner):
    def __init__(self) -> None:
        with open('distributions_standardized.json', 'r') as fp:
            self.distributions = json.load(fp)

    def linkSimulator(self, simulator):
        self.problem = simulator.problem
        distributions_or = simulator.problem.processing_time_distribution
        # Initialize the transformed dictionary
        self.distributions = {}

        # Iterate over the input dictionary
        for (main_key, sub_key), (first_value, _) in distributions_or.items():
            if main_key not in self.distributions:
                self.distributions[main_key] = {}
            self.distributions[main_key][sub_key] = first_value

        # standardize the distributions
        for key in self.distributions.keys():
            if key == 'Turning Rework':
                print(self.distributions[key])
            values = list(self.distributions[key].values())
            mean = np.mean(values)
            std = np.std(values)
            for sub_key in self.distributions[key].keys():
                if std != 0:
                    self.distributions[key][sub_key] = (self.distributions[key][sub_key] - mean) / std
                else:
                    self.distributions[key][sub_key] = mean

    def get_possible_assignments(self, available_resources, unassigned_tasks, resource_pool):
        possible_assignments = []
        for task in unassigned_tasks:
            for resource in available_resources:
                if resource in resource_pool[task.task_type]:
                    possible_assignments.append((resource, task))
        return possible_assignments

    def plan(self, available_resources, unassigned_tasks, resource_pool):
        available_resources = available_resources.copy()
        unassigned_tasks = unassigned_tasks.copy()
        assignments = []

        possible_assignments = self.get_possible_assignments(available_resources, unassigned_tasks, resource_pool)

        while len(possible_assignments) > 0:

            spt = 999999
            best_assignment = None
            for assignment in possible_assignments:  # assignment[0] = task, assignment[1]= resource
                if self.distributions[assignment[1].task_type][assignment[0]] < spt:
                    spt = self.distributions[assignment[1].task_type][assignment[0]]
                    best_assignment = assignment

            # check if best assignment is None
            if best_assignment is None:
                break
            unassigned_tasks.remove(best_assignment[1])
            available_resources.remove(best_assignment[0])
            best_assignment = (best_assignment[0], best_assignment[1].task_type)
            assignments.append(best_assignment)

            possible_assignments = self.get_possible_assignments(available_resources, unassigned_tasks, resource_pool)

        return assignments

    def report(self, event):
        pass  # print(event)


class ShortestProcessingTime(Planner):
    def __init__(self) -> None:
        # with open('distributions.json', 'r') as fp:
        #    self.distributions = json.load(fp)
        self.distributions = None

    def linkSimulator(self, simulator):
        self.problem = simulator.problem
        self.simulator = simulator
        distributions_or = simulator.problem.processing_time_distribution
        # Initialize the transformed dictionary
        self.distributions = {}

        # Iterate over the input dictionary
        for (main_key, sub_key), (first_value, _) in distributions_or.items():
            if main_key not in self.distributions:
                self.distributions[main_key] = {}
            self.distributions[main_key][sub_key] = first_value

    def get_possible_assignments(self, available_resources, unassigned_tasks, resource_pool):
        possible_assignments = []
        for task in unassigned_tasks:
            for resource in available_resources:
                if type(task) != str:
                    task = task.task_type
                if resource in resource_pool[task]:
                    possible_assignments.append((resource, task))
        return possible_assignments

    def get_possible_assignments_original(self, available_resources, unassigned_tasks, resource_pool):
        possible_assignments = []
        for task in unassigned_tasks:
            for resource in available_resources:
                if resource in resource_pool[task.task_type]:
                    possible_assignments.append((resource, task))
        return possible_assignments

    def plan(self, available_resources, unassigned_tasks, resource_pool):
        available_resources = available_resources.copy()
        unassigned_tasks = unassigned_tasks.copy()
        assignments = []

        possible_assignments = self.get_possible_assignments_original(available_resources, unassigned_tasks,
                                                                      resource_pool)
        while len(possible_assignments) > 0:
            spt = 999999
            for assignment in possible_assignments:  # assignment[0] = task, assignment[1]= resource
                if self.distributions[assignment[1].task_type][assignment[0]] < spt:
                    spt = self.distributions[assignment[1].task_type][assignment[0]]
                    best_assignment = assignment

            unassigned_tasks.remove(best_assignment[1])
            available_resources.remove(best_assignment[0])
            best_assignment = (best_assignment[0], best_assignment[1].task_type)
            assignments.append(best_assignment)
            possible_assignments = self.get_possible_assignments_original(available_resources, unassigned_tasks,
                                                                          resource_pool)
        return assignments

    def report(self, event):
        pass  # print(event)

    def plan_from_trace(self, available_resources, unassigned_tasks, resource_pool):
        available_resources = available_resources.copy()
        unassigned_tasks = unassigned_tasks.copy()
        assignments = []

        possible_assignments = self.get_possible_assignments(available_resources, unassigned_tasks, resource_pool)
        while len(possible_assignments) > 0:
            spt = 999999
            for assignment in possible_assignments:  # assignment[0] = task, assignment[1]= resource
                if self.distributions[assignment[1]][assignment[0]] < spt:
                    spt = self.distributions[assignment[1]][assignment[0]]
                    best_assignment = assignment

            unassigned_tasks.remove(best_assignment[1])
            available_resources.remove(best_assignment[0])
            best_assignment = (best_assignment[0], best_assignment[1])
            assignments.append(best_assignment)
            possible_assignments = self.get_possible_assignments(available_resources, unassigned_tasks, resource_pool)
        return assignments


class FIFOProcess(Planner):
    def __str__(self) -> str:
        return 'FIFO'

    def __init__(self):
        self.resource_pools = None  # passed through simulator
        self.task_types = None

    def get_possible_assignments(self, available_resources, unassigned_tasks, resource_pool):
        possible_assignments = []
        for task in unassigned_tasks:
            for resource in available_resources:
                if resource in resource_pool[task.task_type]:
                    possible_assignments.append((resource, task))
        return possible_assignments

    def plan(self, available_resources, available_tasks, resource_pools):
        available_tasks = available_tasks.copy()
        available_resources = available_resources.copy()
        self.task_types = list(resource_pools.keys())

        assignments = []
        case_priority_order = sorted(list(set([task.case_id for task in
                                               available_tasks])))  # cases are characterized by a monotonically increasing case_id
        priority_case = 0
        possible_assignments = self.get_possible_assignments(available_resources, available_tasks, resource_pools)
        while len(possible_assignments) > 0:
            priority_task_types = [task.task_type for task in available_tasks if
                                   task.case_id == case_priority_order[priority_case]]

            best_assignments = []
            while len(best_assignments) == 0:
                for possible_assignment in possible_assignments:
                    if possible_assignment[1].task_type in priority_task_types:
                        best_assignments.append((possible_assignment[0], possible_assignment[1].task_type))
                        possible_assignments.remove(possible_assignment)
                if len(best_assignments) == 0:
                    priority_case += 1
                    priority_task_types = [task.task_type for task in available_tasks if
                                           task.case_id == case_priority_order[priority_case]]

        return best_assignments


class FIFOActivity(Planner):
    def __str__(self) -> str:
        return 'FIFO'

    def __init__(self):
        self.resource_pools = None  # passed through simulator
        self.task_types = None

    def get_possible_assignments(self, available_resources, unassigned_tasks, resource_pool):
        possible_assignments = []
        for task in unassigned_tasks:
            for resource in available_resources:
                if resource in resource_pool[task.task_type]:
                    possible_assignments.append((resource, task))
        return possible_assignments

    def plan(self, available_resources, available_tasks, resource_pools):
        pass


class PPOPlannerTianshou(Planner):
    """A :class:`.Planner` that assigns tasks to resources following policy dictated by (pretrained) DRL algorithm."""

    def __init__(self, policy, preprocess_fn=None, actor_network_name=None) -> None:

        self.policy = policy
        self.policy.load_state_dict(state_dict=torch.load(f"./{actor_network_name}.pt"))  # ('ppo_graph.pt'))

        self.resources = None
        self.task_types = None
        self.inputs = None
        self.output = []
        self.resource_pools_indexes = {}
        self.preprocess_fn = preprocess_fn

        self.simulator = None

    def getState(self, available_resources, unassigned_tasks, busy_resources):
        return self.simulator.get_graph_state()

    def get_state_from_trace(self, available_resources, unassigned_tasks, busy_resources, mask):
        # Create the graph
        graph = {}

        # Add node types
        # graph['edge_index'] = np.array([[], []], dtype=np.int64)
        # graph['edge_attr'] = np.array([], dtype=np.float32)
        # graph['x'] = np.array([], dtype=np.float32)
        # graph['y'] = np.array([], dtype=np.float32)
        graph['global_attr'] = np.array([len(self.simulator.resources)], dtype=np.float32)

        # Add resource nodes
        resources_available = np.isin(self.simulator.resources, list(available_resources)).astype(int)
        resources_busy_time = np.zeros(len(self.simulator.resources))  # placeholder
        resources_x = np.column_stack([resources_available, resources_busy_time])
        graph['resources'] = resources_x

        graph['assignments'] = np.expand_dims(np.array(self.simulator.assignment_nodes_attr, dtype=np.float32), axis=1)

        task_types_num = np.array(
            [len([t for t in unassigned_tasks if t == task_type]) for task_type in self.simulator.task_types])
        if task_types_num.sum() != 0:
            task_types_num = task_types_num / task_types_num.sum()

        graph['task_types'] = task_types_num

        # graph[str(('resources', 'edge', 'assignments'))] = np.array(resource_to_assignment_edges, dtype=np.int64)

        # graph[str(('task_types', 'edge', 'assignments'))] = np.array(task_type_to_assignment_edges, dtype=np.int64)

        graph['mask'] = mask

        graph['reconstruct_edges'] = np.array(self.simulator.edge_index)

        return {'graph_dict': graph}

    def plan_from_trace(self, available_resources, unassigned_tasks, busy_resources, mask):

        assignments = []

        while self.available_assignments(unassigned_tasks, available_resources):

            if self.preprocess_fn is not None:
                obs = self.preprocess_fn(
                    **self.get_state_from_trace(available_resources, unassigned_tasks, busy_resources, mask))
            else:
                obs = self.get_state_from_trace(available_resources, unassigned_tasks, busy_resources, mask)

            action, state_ = self.policy.actor([obs['graph_dict']])
            action = action.argmax().item()

            if self.simulator.output[action] == 'Postpone':
                print("POSTPONED")
                return "Postpone"  # no assignment
            else:
                # print(self.simulator.output[action])
                resource, task = self.take_action(action)

            assignment = (resource, (next(x for x in unassigned_tasks if x == task)))
            # print(f"Assigning resource {assignment[0]} to task {assignment[1]}")

            available_resources.remove(assignment[0])
            unassigned_tasks.remove(assignment[1])

            unassigned_task_types_num = {task_type: len([t for t in unassigned_tasks if t == task_type]) for task_type
                                         in self.simulator.task_types}

            mask = np.array(
                [0 if resource not in available_resources or unassigned_task_types_num[task_type] == 0 else 1
                 for resource in self.simulator.resources for task_type in self.simulator.task_types
                 if resource in self.simulator.problem.resource_pools[task_type]], dtype=np.float32)

            assignment = (assignment[0], assignment[1])

            assignments.append(assignment)
        return assignments

    def getActionMasks(self, state):
        mask = [0 for _ in range(len(self.simulator.output))]

        for task_type in self.simulator.task_types:
            if state[self.simulator.input.index(task_type)] > 0:
                for resource in self.simulator.problem.resource_pools[task_type]:
                    if state[self.simulator.input.index(resource + '_availability')] > 0:
                        mask[self.simulator.output.index((task_type, resource))] = 1

        mask[-1] = 1  # Set postpone action to 1

        return list(map(bool, mask))

    # pass the simulator for bidirectional communication
    def linkSimulator(self, simulator):
        self.simulator = simulator
        self.problem = simulator.problem

    def take_action(self, action):
        return self.simulator.output[action]

    def available_assignments(self, unassigned_tasks, available_resources):
        for task in unassigned_tasks:
            # import pdb; pdb.set_trace()
            if type(task) != str:
                task = task.task_type
            if len(set(available_resources).intersection(
                    set(self.simulator.problem_resource_pool[task]))) > 0:
                return True
        return False

    def plan(self, available_resources, unassigned_tasks, resource_pool):

        assignments = []

        available_resources = copy.deepcopy(self.simulator.available_resources)
        unassigned_tasks = copy.deepcopy(list(self.simulator.unassigned_tasks.values()))
        busy_resources = copy.deepcopy(self.simulator.busy_resources)

        while self.available_assignments(unassigned_tasks, available_resources) and not self.simulator.postponed:

            if self.preprocess_fn is not None:
                obs = self.preprocess_fn(**self.get_current_obs(available_resources, unassigned_tasks, busy_resources))
            else:
                obs = self.get_current_obs(available_resources, unassigned_tasks, busy_resources)

            action, state_ = self.policy.actor([obs['graph_dict']])
            action = action.argmax().item()

            if self.simulator.output[action] == 'Postpone':
                # print("POSTPONED")
                self.simulator.postponed = True
                return ["Postpone"]  # no assignment
            else:
                # print(self.simulator.output[action])
                resource, task = self.take_action(action)

                assignment = (resource, (next(x for x in unassigned_tasks if x.task_type == task)))
                # print(f"Assigning resource {assignment[0]} to task {assignment[1]}")

                available_resources.remove(assignment[0])
                unassigned_tasks.remove(assignment[1])
                busy_resources[assignment[0]] = (assignment[1], self.simulator.now)

                assignment = (assignment[0], assignment[1].task_type)

                assignments.append(assignment)
        return assignments

    def get_current_obs(self, available_resources, unassigned_tasks, busy_resources):
        return self.simulator.get_graph_state_from_lists(available_resources, unassigned_tasks, busy_resources)

    def report(self, event):
        pass  # print(event)
