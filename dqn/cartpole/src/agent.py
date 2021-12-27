from .replay_buffer import ReplayBuffer, StateTransition
from .dqn import DQN
import torch
import torch.nn.functional as F
import random
import math
import gym


class Agent(object):
    def __init__(self,
                 env: gym.wrappers.time_limit.TimeLimit,
                 batch_size: int,
                 gamma: float,
                 eps_start: float,
                 eps_end: float,
                 eps_decay: float,
                 target_update: int,
                 num_actions: int,
                 observation_size: int,
                 hidden_size: int) -> None:
        """
        This class contains all the logic necessary to train an agent that can solve the cartpole gym environment
        Args:
            env: cartpole gym environment instance
            batch_size: number of samples processed during model update
            gamma: reward discount factor
            eps_start: starting probability of choosing to explore
            eps_end: ending probability of choosing to explore
            eps_decay: exploration probability decay
            target_update: frequency of target network update
            num_actions: size of action space
            observation_size: size of observation
            hidden_size: number of neurons in hidden layers
        """
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.num_actions = num_actions
        self.observation_size = observation_size
        self.hidden_size = hidden_size

        self.memory = ReplayBuffer(10000)

        self.policy_net = DQN(self.observation_size, self.num_actions, self.hidden_size)
        self.target_net = DQN(self.observation_size, self.num_actions, self.hidden_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.steps_done = 0

    def __save_agent(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)

    def __load_agent(self, path: str) -> None:
        self.policy_net = DQN(self.observation_size, self.num_actions, self.hidden_size)
        self.policy_net.load_state_dict(torch.load(path))
        self.policy_net.eval()

    def __select_action(self, observation: torch.Tensor) -> torch.Tensor:
        sample = random.random()
        # exploration threshold that decays over time (number of env steps)
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_end)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return torch.tensor([self.policy_net(observation).argmax()]).unsqueeze(1)
        else:
            return torch.tensor([[random.randrange(self.num_actions)]], dtype=torch.long)

    def __optimize_agent(self) -> None:

        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)

        # convert array of StateTransitions to StateTransition of arrays.
        batch = StateTransition(*zip(*transitions))

        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action).squeeze(1)
        reward_batch = torch.stack(batch.reward).squeeze(1)
        next_states_batch = torch.stack(batch.next_state)

        state_values = self.policy_net(state_batch)
        next_state_values = self.target_net(next_states_batch)

        # mask for updating only q values that correspond to actions
        mask = F.one_hot(action_batch, num_classes=self.num_actions).bool().squeeze(1)

        state_values[mask] = (next_state_values.max(1)[0] * self.gamma) + reward_batch

        self.policy_net.backward(state_batch, state_values)

    def train_agent(self, model_save_path: str, num_episodes: int, goal_score: int, early_stop_threshold: int) -> None:
        """
        This method trains the agent and saves the weights to specified directory
        Args:
            model_save_path: path for saving model in pkl format
            num_episodes: total number of episodes that agent will train for
            goal_score: minimum score that agents need to achieve
            early_stop_threshold: number of consecutive occurrences of goal_score needed for early stopping

        Returns:
            None
        """
        goal_score_reached = 0

        for i_episode in range(num_episodes):
            episode_reward = 0
            state = torch.tensor(self.env.reset())
            episode_done = False
            while not episode_done:
                # select and perform an action
                action = self.__select_action(state)

                next_state, reward, episode_done, _ = self.env.step(action.item())

                if episode_done:
                    reward = -1

                episode_reward += reward

                reward = torch.tensor([reward])
                next_state = torch.tensor(next_state)

                # store the state transition in memory
                self.memory.push(state, action, next_state, reward)

                # move to the next state
                state = next_state

                # perform one step of the optimization (on the policy network)
                self.__optimize_agent()

            print(f'Episode: {i_episode}, reward: {episode_reward}')
            # update the target network, copying all weights and biases in DQN
            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # if score achieved in finished episode is goal score increment its consecutive occurrence counter
            if episode_reward >= goal_score:
                goal_score_reached += 1
            else:
                goal_score_reached = 0

            # early stop condition - as soon as model reaches desired score for n consecutive iterations
            if goal_score_reached == early_stop_threshold:
                break

        self.__save_agent(model_save_path)

    def test_agent(self, num_episodes: int, model_load_path: str) -> None:
        """
        This method evaluates the agent by acting in the environment
        Args:
            num_episodes: number of evaluation episodes that agent will run
            model_load_path: path to saved model generated by the train_model method

        Returns:
            None
        """
        self.__load_agent(model_load_path)
        episode_rewards = []
        for i_episode in range(num_episodes):
            episode_done = False
            state = torch.tensor(self.env.reset())
            episode_reward = 0
            while not episode_done:
                action = self.__select_action(state)
                next_state, reward, episode_done, _ = self.env.step(action.item())
                next_state = torch.tensor(next_state)
                state = next_state
                episode_reward += reward
            episode_rewards.append(episode_reward)

        print(sum(episode_rewards) / num_episodes)
        print(episode_rewards)
