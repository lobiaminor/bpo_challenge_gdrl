import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List

from simulator import Simulator, MinedProblem


class BPOEnv(gym.Env):
    def __init__(self, render_mode='human', action_mode="edge_selection",
                 instance_file="./BPI Challenge 2017 - instance.pickle", running_time=365 * 24, interarrival_rate_multiplier = 1) -> None:
        super().__init__()
        self.num_envs = 1
        self.instance_file = instance_file
        self.running_time = running_time
        self.counter = 0
        self.nr_postpone = 0

        self.instance_file = instance_file
        self.running_time = running_time
        self.counter = 0
        self.nr_postpone = 0
        self.render_mode = render_mode
        self.action_mode = action_mode
        self.minimal_simulator = False

        self.problem = MinedProblem.from_file(instance_file)

        self.simulator = Simulator(running_time=self.running_time, problem=self.problem, report=False,
                                   instance_file=self.instance_file, planner=None, interarrival_rate_multiplier=interarrival_rate_multiplier)

        self.observation_space = spaces.Dict(
            {
                "resources": spaces.Box(low=0, high=1, shape=(len(self.simulator.resources), 2), dtype=np.float64),
                "task_types": spaces.Box(low=0, high=1, shape=(len(self.simulator.task_types), 1), dtype=np.float64),
                "edge_index": spaces.Box(low=0, high=len(self.simulator.resources),
                                         shape=(2, len(self.simulator.output)), dtype=np.float64),
                "edge_attr": spaces.Box(low=0, high=1, shape=(len(self.simulator.output), 1), dtype=np.float64),
            }
        )

        # spaces.Discrete returns a number between 0 and len(self.simulator.output)
        self.action_space = spaces.Discrete(
            len(self.simulator.output))  # action space is the cartesian product of tasks and resources in their resource pool

        #while not self.simulator.available_assignments():
        #    self.simulator.run()  # Run the simulator to get to the first decision

        # used to reduce the calls to get_graph_state
        #self.current_state = self.simulator.get_graph_state()['graph_dict']

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        # Assign one resources per iteration. If possible, another is assigned in next step without advancing simulator
        assignment = self.simulator.output[action]

        if assignment == 'Postpone':
            self.nr_postpone += 1
            print("Postponing at time", self.simulator.now, "Total postpones:", self.nr_postpone)
        self.counter += 1
        if self.counter % 1000 == 0:
            print('Postponed:', f'{self.nr_postpone}/1000')
            if len(self.simulator.unassigned_tasks) > 0:
                print(
                    [sum([1 if task.task_type == el else 0 for task in self.simulator.unassigned_tasks.values()]) for el
                     in self.simulator.task_types], '\n')
            self.nr_postpone = 0

        if assignment != 'Postpone':
            # print('stuck 1', assignment, self.simulator.now)
            self.simulator.schedule_resources([assignment])

        else:
            pass

        # While assignment not possible and simulator not finished (postpone always possible)
        while (not self.simulator.available_assignments()) and (self.simulator.status != 'FINISHED'):
            # print('ASSIGNED', self.simulator.now)
            self.simulator.run()  # breaks each time at resource assignment, continues if no assignment possible

        reward = -self.simulator.current_reward

        # Simulation is finished, return current reward (with penalties)
        if self.simulator.status == 'FINISHED':
            print('FINAL REWARD', -self.simulator.current_reward)
            print('Tot steps: ', self.counter)
            self.counter = 0
            self.current_state = self.simulator.get_graph_state()['graph_dict']
            return (
                self.current_state,
                reward,
                True,
                False,
                {}
            )
        else:


            self.current_state = self.simulator.get_graph_state()['graph_dict']
            return (
                self.current_state,
                reward,
                False,
                False,
                {}
            )

    def reset(self, seed=None):
        """Resets the environment to an initial state and returns an initial
        observation.

        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.

        Returns:
            observation (object): the initial observation.
        """

        print("-------Resetting environment-------")
        self.simulator.reset_simulator()
        while not self.simulator.available_assignments():
            self.simulator.run()
        self.current_state = self.simulator.get_graph_state()['graph_dict']
        return self.current_state, dict()

    def render(self, mode='human', close=False):
        pass

    def define_action_masks(self) -> List[bool]:
        # the mask is a list of booleans, where the i-th element is True if the i-th action is valid
        # here actions correspond to selecting an assignment node in the graph

        mask = self.current_state['edge_attr']
        return list(map(bool, mask))

    def action_masks(self) -> List[bool]:
        return self.define_action_masks()

    def seed(self, seed):
        return np.random.seed(seed)
