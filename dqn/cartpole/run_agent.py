import gym
from src.agent import Agent

env = gym.make('CartPole-v1')

BATCH_SIZE = 256
GAMMA = 0.995
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 200
TARGET_UPDATE = 10
MODEL_SAVE_PATH = './artifacts/model/model.pkl'
NUM_ACTIONS = env.action_space.n
OBSERVATION_SIZE = len(env.reset())
HIDDEN_SIZE = 64
NUM_TRAIN_EPISODES = 1000
# 500 is the perfect score, but because reward for final episode action (the one that returns done = True flag) is set
# to -1 during training, it's impossible to get a reward cumulative episode reward above 498
TRAIN_GOAL_SCORE = 498
EARLY_STOP_THRESHOLD = 5
NUM_TEST_EPISODES = 50

agent = Agent(env, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TARGET_UPDATE,
              NUM_ACTIONS, OBSERVATION_SIZE, HIDDEN_SIZE)

agent.train_agent(MODEL_SAVE_PATH, NUM_TRAIN_EPISODES, TRAIN_GOAL_SCORE, EARLY_STOP_THRESHOLD)
agent.test_agent(NUM_TEST_EPISODES, MODEL_SAVE_PATH)