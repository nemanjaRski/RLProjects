import random
import torch
from collections import deque, namedtuple
from typing import List

StateTransition = namedtuple('StateTransition', ('state', 'action', 'next_state', 'reward'))


class ReplayBuffer(object):
    def __init__(self, capacity: int) -> None:
        self.buffer = deque([], maxlen=capacity)

    def push(self,
             state: torch.Tensor,
             action: torch.Tensor,
             next_state: torch.Tensor,
             reward: torch.Tensor) -> None:
        self.buffer.append(StateTransition(state, action, next_state, reward))

    def sample(self, batch_size) -> List[StateTransition]:
        return random.sample(self.buffer, batch_size)

    def empty(self) -> None:
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)
    