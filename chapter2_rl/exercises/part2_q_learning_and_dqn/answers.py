#%%
import os
from dataclasses import dataclass
from typing import Optional, Union, List
import numpy as np
import gym
import gym.spaces
import gym.envs.registration
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm, trange
import sys
import time
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Union, Tuple
import torch as t
from torch import nn, Tensor
from gym.spaces import Discrete, Box
from numpy.random import Generator
import pandas as pd
import wandb
import pandas as pd
from pathlib import Path
from jaxtyping import Float, Int, Bool
from IPython.display import clear_output
import warnings
warnings.filterwarnings('ignore')

Arr = np.ndarray

# Make sure exercises are in the path
chapter = r"chapter2_rl"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part2_dqn"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from part1_intro_to_rl.utils import make_env
from part1_intro_to_rl.solutions import Environment, Toy, Norvig, find_optimal_policy
import part2_q_learning_and_dqn.utils as utils
import part2_q_learning_and_dqn.tests as tests
from plotly_utils import line, cliffwalk_imshow, plot_cartpole_obs_and_dones

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"
# %%
ObsType = int
ActType = int

class DiscreteEnviroGym(gym.Env):
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Discrete
    '''
    A discrete environment class for reinforcement learning, compatible with OpenAI Gym.

    This class represents a discrete environment where actions and observations are discrete.
    It is designed to interface with a provided `Environment` object which defines the 
    underlying dynamics, states, and actions.

    Attributes:
        action_space (gym.spaces.Discrete): The space of possible actions.
        observation_space (gym.spaces.Discrete): The space of possible observations (states).
        env (Environment): The underlying environment with its own dynamics and properties.
    '''
    def __init__(self, env: Environment):
        super().__init__()
        self.env = env
        self.observation_space = gym.spaces.Discrete(env.num_states)
        self.action_space = gym.spaces.Discrete(env.num_actions)
        self.reset()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        '''
        Execute an action and return the new state, reward, done flag, and additional info.
        The behaviour of this function depends primarily on the dynamics of the underlying
        environment.
        '''
        (states, rewards, probs) = self.env.dynamics(self.pos, action)
        idx = self.np_random.choice(len(states), p=probs)
        (new_state, reward) = (states[idx], rewards[idx])
        self.pos = new_state
        done = self.pos in self.env.terminal
        return (new_state, reward, done, {"env": self.env})

    def reset(self, seed: Optional[int] = None, options=None) -> ObsType:
        '''
        Resets the environment to its initial state.
        '''
        super().reset(seed=seed)
        self.pos = self.env.start
        return self.pos

    def render(self, mode="human"):
        assert mode == "human", f"Mode {mode} not supported!"
# %%
gym.envs.registration.register(
    id="NorvigGrid-v0",
    entry_point=DiscreteEnviroGym,
    max_episode_steps=100,
    nondeterministic=True,
    kwargs={"env": Norvig(penalty=-0.04)},
)

gym.envs.registration.register(
    id="ToyGym-v0",
    entry_point=DiscreteEnviroGym,
    max_episode_steps=2,
    nondeterministic=False,
    kwargs={"env": Toy()}
)
# %%
@dataclass
class Experience:
    '''
    A class for storing one piece of experience during an episode run.
    '''
    obs: ObsType
    act: ActType
    reward: float
    new_obs: ObsType
    new_act: Optional[ActType] = None


@dataclass
class AgentConfig:
    '''Hyperparameters for agents'''
    epsilon: float = 0.1
    lr: float = 0.05
    optimism: float = 0

defaultConfig = AgentConfig()


class Agent:
    '''Base class for agents interacting with an environment (you do not need to add any implementation here)'''
    rng: np.random.Generator

    def __init__(self, env: DiscreteEnviroGym, config: AgentConfig = defaultConfig, gamma: float = 0.99, seed: int = 0):
        print(seed)
        self.env = env
        self.reset(seed)
        self.config = config
        self.gamma = gamma
        self.num_actions = env.action_space.n
        self.num_states = env.observation_space.n
        self.name = type(self).__name__

    def get_action(self, obs: ObsType) -> ActType:
        raise NotImplementedError()

    def observe(self, exp: Experience) -> None:
        '''
        Agent observes experience, and updates model as appropriate.
        Implementation depends on type of agent.
        '''
        pass

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def run_episode(self, seed) -> List[int]:
        '''
        Simulates one episode of interaction, agent learns as appropriate
        Inputs:
            seed : Seed for the random number generator
        Outputs:
            The rewards obtained during the episode
        '''
        rewards = []
        obs = self.env.reset(seed=seed)
        self.reset(seed=seed)
        done = False
        while not done:
            act = self.get_action(obs)
            (new_obs, reward, done, info) = self.env.step(act)
            exp = Experience(obs, act, reward, new_obs)
            self.observe(exp)
            rewards.append(reward)
            obs = new_obs
        return rewards

    def train(self, n_runs=500):
        '''
        Run a batch of episodes, and return the total reward obtained per episode
        Inputs:
            n_runs : The number of episodes to simulate
        Outputs:
            The discounted sum of rewards obtained for each episode
        '''
        all_rewards = []
        for seed in trange(n_runs):
            rewards = self.run_episode(seed)
            all_rewards.append(utils.sum_rewards(rewards, self.gamma))
        return all_rewards


class Random(Agent):
    def get_action(self, obs: ObsType) -> ActType:
        return self.rng.integers(0, self.num_actions)
# %%
class Cheater(Agent):
    def __init__(self, env: DiscreteEnviroGym, config: AgentConfig = defaultConfig, gamma=0.99, seed=0):
        super().__init__(env, config, gamma, seed)
        self.optimal_policy = find_optimal_policy(env=env.unwrapped.env, gamma=gamma)

    def get_action(self, obs):
        return self.optimal_policy[obs]


env_toy = gym.make("ToyGym-v0")
agents_toy: List[Agent] = [Cheater(env_toy), Random(env_toy)]
returns_list = []
names_list = []
for agent in agents_toy:
    returns = agent.train(n_runs=100)
    returns_list.append(utils.cummean(returns))
    names_list.append(agent.name)

line(
    returns_list,
    names=names_list,
    title=f"Avg. reward on {env_toy.spec.name}",
    labels={"x": "Episode", "y": "Avg. reward", "variable": "Agent"},
    template="simple_white", width=700, height=400,
)
#%%
class EpsilonGreedy(Agent):
    '''
    A class for SARSA and Q-Learning to inherit from.
    '''
    def __init__(self, env: DiscreteEnviroGym, config: AgentConfig = defaultConfig, gamma: float = 0.99, seed: int = 0):
        super().__init__(env, config, gamma, seed)
        self.Q = np.zeros((self.num_states, self.num_actions)) + self.config.optimism

    def get_action(self, obs: ObsType) -> ActType:
        '''
        Selects an action using epsilon-greedy with respect to Q-value estimates
        '''
        if self.rng.random() < self.config.epsilon:
            action = self.rng.integers(0, self.num_actions)
        else:
            action = np.argmax(self.Q[obs]).item()
        return action


class QLearning(EpsilonGreedy):
    def observe(self, exp: Experience) -> None:
        self.Q[exp.obs, exp.act] += self.config.lr * (exp.reward + self.gamma * np.max(self.Q[exp.new_obs]).item() - self.Q[exp.obs, exp.act])


class SARSA(EpsilonGreedy):
    def observe(self, exp: Experience):
        self.Q[exp.obs, exp.act] += self.config.lr * (exp.reward + self.gamma * self.Q[exp.new_obs, exp.new_act] - self.Q[exp.obs, exp.act])

    def run_episode(self, seed) -> List[int]:
        rewards = []
        obs = self.env.reset(seed=seed)
        act = self.get_action(obs)
        self.reset(seed=seed)
        done = False
        while not done:
            (new_obs, reward, done, info) = self.env.step(act)
            new_act = self.get_action(new_obs)
            exp = Experience(obs, act, reward, new_obs, new_act)
            self.observe(exp)
            rewards.append(reward)
            obs = new_obs
            act = new_act
        return rewards


n_runs = 1000
gamma = 0.99
seed = 1
env_norvig = gym.make("NorvigGrid-v0")
config_norvig = AgentConfig(epsilon=0.01, lr=0.1, optimism=-3)
args_norvig = (env_norvig, config_norvig, gamma, seed)
agents_norvig: List[Agent] = [Cheater(*args_norvig), QLearning(*args_norvig), SARSA(*args_norvig), Random(*args_norvig)]
returns_norvig = {}
fig = go.Figure(layout=dict(
    title_text=f"Avg. reward on {env_norvig.spec.name}",
    template="simple_white",
    xaxis_range=[-30, n_runs+30],
    width=700, height=400,
))
for agent in agents_norvig:
    returns = agent.train(n_runs)
    fig.add_trace(go.Scatter(y=utils.cummean(returns), name=agent.name))
fig.show()
# %%
gamma = 1
seed = 0

config_cliff = AgentConfig(epsilon=0.1, lr = 0.1, optimism=0)
env = gym.make("CliffWalking-v0")
n_runs = 2500
args_cliff = (env, config_cliff, gamma, seed)

returns_list = []
name_list = []
agents: List[Union[QLearning, SARSA]] = [QLearning(*args_cliff), SARSA(*args_cliff)]

for agent in agents:
    returns = agent.train(n_runs)[1:]
    returns_list.append(utils.cummean(returns))
    name_list.append(agent.name)
    V = agent.Q.max(axis=-1).reshape(4, 12)
    pi = agent.Q.argmax(axis=-1).reshape(4, 12)
    cliffwalk_imshow(V, pi, title=f"CliffWalking: {agent.name} Agent", width=800, height=400)

line(
    returns_list,
    names=name_list,
    template="simple_white",
    title="Q-Learning vs SARSA on CliffWalking-v0",
    labels={"x": "Episode", "y": "Avg. reward", "variable": "Agent"},
    width=700, height=400,
)
# %%
class QNetwork(nn.Module):
    '''For consistency with your tests, please wrap your modules in a `nn.Sequential` called `layers`.'''
    layers: nn.Sequential

    def __init__(
        self,
        dim_observation: int,
        num_actions: int,
        hidden_sizes: List[int] = [120, 84]
    ):
        super().__init__()
        self.dim_observation = dim_observation
        self.num_actions = num_actions
        self.hidden_sizes = hidden_sizes
        self.layers = nn.Sequential(
            nn.Linear(dim_observation, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_actions)
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.layers(x)


net = QNetwork(dim_observation=4, num_actions=2)
n_params = sum((p.nelement() for p in net.parameters()))
assert isinstance(getattr(net, "layers", None), nn.Sequential)
print(net)
print(f"Total number of parameters: {n_params}")
print("You should manually verify network is Linear-ReLU-Linear-ReLU-Linear")
assert n_params == 10934
# %%
@dataclass
class ReplayBufferSamples:
    '''
    Samples from the replay buffer, converted to PyTorch for use in neural network training.

    Data is equivalent to (s_t, a_t, r_{t+1}, d_{t+1}, s_{t+1}).
    '''
    observations: Tensor # shape [sample_size, *observation_shape]
    actions: Tensor # shape [sample_size, *action_shape]
    rewards: Tensor # shape [sample_size,]
    dones: Tensor # shape [sample_size,]
    next_observations: Tensor # shape [sample_size, observation_shape]

    def __post_init__(self):
        for exp in self.__dict__.values():
            assert isinstance(exp, Tensor), f"Error: expected type tensor, found {type(exp)}"


class ReplayBuffer:
    '''
    Contains buffer; has a method to sample from it to return a ReplayBufferSamples object.
    '''
    rng: Generator
    observations: np.ndarray # shape [buffer_size, *observation_shape]
    actions: np.ndarray # shape [buffer_size, *action_shape]
    rewards: np.ndarray # shape [buffer_size,]
    dones: np.ndarray # shape [buffer_size,]
    next_observations: np.ndarray # shape [buffer_size, *observation_shape]

    def __init__(self, num_environments: int, obs_shape: Tuple[int], action_shape: Tuple[int], buffer_size: int, seed: int):
        assert num_environments == 1, "This buffer only supports SyncVectorEnv with 1 environment inside."
        self.num_environments = num_environments
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.buffer_size = buffer_size
        self.rng = np.random.default_rng(seed)

        self.observations = np.empty((0, *self.obs_shape), dtype=np.float32)
        self.actions = np.empty(0, dtype=np.int32)
        self.rewards = np.empty(0, dtype=np.float32)
        self.dones = np.empty(0, dtype=bool)
        self.next_observations = np.empty((0, *self.obs_shape), dtype=np.float32)


    def add(
        self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray, next_obs: np.ndarray
    ) -> None:
        '''
        obs: shape (num_environments, *observation_shape)
            Observation before the action
        actions: shape (num_environments, *action_shape)
            Action chosen by the agent
        rewards: shape (num_environments,)
            Reward after the action
        dones: shape (num_environments,)
            If True, the episode ended and was reset automatically
        next_obs: shape (num_environments, *observation_shape)
            Observation after the action
            If done is True, this should be the terminal observation, NOT the first observation of the next episode.
        '''
        assert obs.shape == (self.num_environments, *self.obs_shape)
        assert actions.shape == (self.num_environments, *self.action_shape)
        assert rewards.shape == (self.num_environments,)
        assert dones.shape == (self.num_environments,)
        assert next_obs.shape == (self.num_environments, *self.obs_shape) 

        self.observations = np.concatenate((self.observations, obs))[-self.buffer_size:]
        self.actions = np.concatenate((self.actions, actions))[-self.buffer_size:]
        self.rewards = np.concatenate((self.rewards, rewards))[-self.buffer_size:]
        self.dones = np.concatenate((self.dones, dones))[-self.buffer_size:]
        self.next_observations = np.concatenate((self.next_observations, next_obs))[-self.buffer_size:]


    def sample(self, sample_size: int, device: t.device) -> ReplayBufferSamples:
        '''
        Uniformly sample sample_size entries from the buffer and convert them to PyTorch tensors on device.
        Sampling is with replacement, and sample_size may be larger than the buffer size.
        '''
        buffer_length = self.observations.shape[0]
        indices = self.rng.integers(0, buffer_length, sample_size)
        return ReplayBufferSamples(
            observations=t.as_tensor(self.observations[indices]).to(device),
            actions=t.as_tensor(self.actions[indices]).to(device),
            rewards=t.as_tensor(self.rewards[indices]).to(device),
            dones=t.as_tensor(self.dones[indices]).to(device),
            next_observations=t.as_tensor(self.next_observations[indices]).to(device),
        )



tests.test_replay_buffer_single(ReplayBuffer)
tests.test_replay_buffer_deterministic(ReplayBuffer)
tests.test_replay_buffer_wraparound(ReplayBuffer)
# %%
rb = ReplayBuffer(num_environments=1, obs_shape=(4,), action_shape=(), buffer_size=256, seed=0)
envs = gym.vector.SyncVectorEnv([make_env("CartPole-v1", 0, 0, False, "test")])
obs = envs.reset()
for i in range(256):
    # Choose a random next action, and take a step in the environment
    actions = envs.action_space.sample()
    (next_obs, rewards, dones, infos) = envs.step(actions)
    # Add observations to buffer, and set obs = next_obs ready for the next step
    rb.add(obs, actions, rewards, dones, next_obs)
    obs = next_obs

plot_cartpole_obs_and_dones(rb.observations, rb.dones, title="CartPole experiences s<sub>t</sub> (dotted lines = termination)")

sample = rb.sample(256, t.device("cpu"))
plot_cartpole_obs_and_dones(sample.observations, sample.dones, title="CartPole experiences s<sub>t</sub> (randomly sampled) (dotted lines = termination)")
# %%
rb = ReplayBuffer(num_environments=1, obs_shape=(4,), action_shape=(), buffer_size=256, seed=0)
envs = gym.vector.SyncVectorEnv([make_env("CartPole-v1", 0, 0, False, "test")])
obs = envs.reset()
for i in range(256):
    # Choose a random next action, and take a step in the environment
    actions = envs.action_space.sample()
    (next_obs, rewards, dones, infos) = envs.step(actions)

    # Get actual next_obs, by replacing next_obs with terminal observation at all envs which are terminated
    real_next_obs = next_obs.copy()
    for environment, done in enumerate(dones):
        if done:
            print(f'Environment {environment} terminated after {infos[0]["episode"]["l"]} steps')
            real_next_obs[environment] = infos[environment]["terminal_observation"]

    # Add the next_obs to the buffer (which has the terminated states), but set obs=new_obs (which has the restarted states)
    rb.add(obs, actions, rewards, dones, real_next_obs)
    obs = next_obs

plot_cartpole_obs_and_dones(rb.next_observations, rb.dones, title="CartPole experiences s<sub>t+1</sub> (dotted lines = termination)")
# %%
def linear_schedule(
    current_step: int, start_e: float, end_e: float, exploration_fraction: float, total_timesteps: int
) -> float:
    '''Return the appropriate epsilon for the current step.

    Epsilon should be start_e at step 0 and decrease linearly to end_e at step (exploration_fraction * total_timesteps).
    In other words, we are in "explore mode" with start_e >= epsilon >= end_e for the first `exploration_fraction` fraction
    of total timesteps, and then stay at end_e for the rest of the episode.
    '''
    if current_step >= exploration_fraction * total_timesteps:
        return end_e
    else:
        return start_e - (start_e - end_e) * (current_step / (exploration_fraction * total_timesteps))


epsilons = [
    linear_schedule(step, start_e=1.0, end_e=0.05, exploration_fraction=0.5, total_timesteps=500)
    for step in range(500)
]
line(epsilons, labels={"x": "steps", "y": "epsilon"}, title="Probability of random action", height=400, width=600)

tests.test_linear_schedule(linear_schedule)
# %%
def epsilon_greedy_policy(
    envs: gym.vector.SyncVectorEnv, q_network: QNetwork, rng: Generator, obs: np.ndarray, epsilon: float
) -> np.ndarray:
    '''With probability epsilon, take a random action. Otherwise, take a greedy action according to the q_network.
    Inputs:
        envs : gym.vector.SyncVectorEnv, the family of environments to run against
        q_network : QNetwork, the network used to approximate the Q-value function
        obs : The current observation
        epsilon : exploration percentage
    Outputs:
        actions: (n_environments, *action_shape) the sampled action for each environment.
    '''
    # Convert `obs` into a tensor so we can feed it into our model
    device = next(q_network.parameters()).device
    obs = t.from_numpy(obs).to(device)

    if rng.random() < epsilon:
        return rng.integers(0, envs.single_action_space.n, (envs.num_envs,))
    else:
        q_values = q_network(obs).detach().numpy()
        return np.argmax(q_values, axis=1)


tests.test_epsilon_greedy_policy(epsilon_greedy_policy)
# %%
@dataclass
class DQNArgs:
    # Basic / global
    seed: int = 1
    cuda: bool = t.cuda.is_available()
    env_id: str = "CartPole-v1"

    # Wandb / logging
    use_wandb: bool = False
    capture_video: bool = True
    exp_name: str = "DQN_implementation"
    log_dir: str = "logs"
    wandb_project_name: str = "CartPoleDQN"
    wandb_entity: Optional[str] = None

    # Duration of different phases
    buffer_size: int = 10_000
    train_frequency: int = 10
    total_timesteps: int = 500_000
    target_network_frequency: int = 500

    # Optimization hyperparameters
    batch_size: int = 128
    learning_rate: float = 0.00025
    start_e: float = 1.0
    end_e: float = 0.1

    # Misc. RL related
    gamma: float = 0.99
    exploration_fraction: float = 0.2

    def __post_init__(self):
        assert self.total_timesteps - self.buffer_size >= self.train_frequency
        self.total_training_steps = (self.total_timesteps - self.buffer_size) // self.train_frequency


args = DQNArgs(batch_size=256)
utils.arg_help(args)
# %%
class DQNAgent:
    '''Base Agent class handling the interaction with the environment.'''

    def __init__(
        self,
        envs: gym.vector.SyncVectorEnv,
        args: DQNArgs,
        rb: ReplayBuffer,
        q_network: QNetwork,
        target_network: QNetwork,
        rng: np.random.Generator
    ):
        self.envs = envs
        self.args = args
        self.rb = rb
        self.next_obs = self.envs.reset() # Need a starting observation!
        self.step = 0
        self.epsilon = args.start_e
        self.q_network = q_network
        self.target_network = target_network
        self.rng = rng

    def play_step(self) -> List[dict]:
        '''
        Carries out a single interaction step between the agent and the environment, and adds results to the replay buffer.

        Returns `infos` (list of dictionaries containing info we will log).
        '''
        obs = self.next_obs
        actions = self.get_actions(obs)
        (next_obs, rewards, dones, infos) = self.envs.step(actions)

        real_next_obs = next_obs.copy()
        for environment, done in enumerate(dones):
            if done:
                print(f'Environment {environment} terminated after {infos[0]["episode"]["l"]} steps')
                real_next_obs[environment] = infos[environment]["terminal_observation"]

        self.rb.add(obs, actions, rewards, dones, real_next_obs)
        self.next_obs = real_next_obs
        self.step += 1

        return infos

    # Add the next_obs to the buffer (which has the terminated states), but set obs=new_obs (which has the restarted states)
    rb.add(obs, actions, rewards, dones, real_next_obs)

    def get_actions(self, obs: np.ndarray) -> np.ndarray:
        '''
        Samples actions according to the epsilon-greedy policy using the linear schedule for epsilon.
        '''
        self.epsilon = linear_schedule(
            current_step=self.step,
            start_e=self.args.start_e,
            end_e=self.args.end_e,
            exploration_fraction=self.args.exploration_fraction,
            total_timesteps=self.args.total_timesteps,
        )
        
        actions = epsilon_greedy_policy(
            envs=self.envs,
            q_network=self.q_network,
            rng=self.rng,
            obs=obs,
            epsilon=self.epsilon,
        )

        return actions


tests.test_agent(DQNAgent)