import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self,
                 observation_size: int,
                 num_actions: int,
                 hidden_size: int,
                 learning_rate: float = 0.001) -> None:

        super(DQN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.LeakyReLU(),
            nn.Linear(2 * hidden_size, num_actions)
        )

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(observation)

    def backward(self, observations: torch.Tensor, y: torch.Tensor) -> None:
        preds = self.model(observations)
        loss = self.loss(preds, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
