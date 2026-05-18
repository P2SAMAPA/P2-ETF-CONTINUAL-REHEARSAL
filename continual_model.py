import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class ContinualLearner:
    def __init__(self, input_dim, hidden_dim=64, lr=1e-3, method='replay', replay_buffer_size=200, ewc_lambda=0.1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MLP(input_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.method = method
        self.ewc_lambda = ewc_lambda
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.fisher = None
        self.optpar = None

    def _compute_fisher(self, X, y):
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(X)
        loss = nn.MSELoss()(outputs.squeeze(), y)
        loss.backward()
        fisher = {name: p.grad.data.clone().pow(2).mean().item() for name, p in self.model.named_parameters() if p.grad is not None}
        return fisher

    def _ewc_loss(self, fisher, optpar):
        loss = 0.0
        for name, p in self.model.named_parameters():
            if name in fisher:
                loss += (fisher[name] * (p - optpar[name]).pow(2)).sum()
        return loss

    def train_step(self, X, y, replay_batch_size=32):
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).to(self.device)

        if self.method == 'replay':
            if len(self.replay_buffer) > 0:
                replay_sample = random.sample(self.replay_buffer, min(replay_batch_size, len(self.replay_buffer)))
                X_replay = torch.stack([s[0] for s in replay_sample]).to(self.device)
                y_replay = torch.stack([s[1] for s in replay_sample]).to(self.device)
                X_combined = torch.cat([X_t, X_replay])
                y_combined = torch.cat([y_t, y_replay])
            else:
                X_combined, y_combined = X_t, y_t
        else:
            X_combined, y_combined = X_t, y_t

        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(X_combined).squeeze()
        loss = nn.MSELoss()(outputs, y_combined)

        if self.method == 'ewc' and self.fisher is not None:
            loss += self.ewc_lambda * self._ewc_loss(self.fisher, self.optpar)

        loss.backward()
        self.optimizer.step()

        if self.method == 'replay':
            for i in range(len(X_t)):
                self.replay_buffer.append((X_t[i].cpu(), y_t[i].cpu()))

        return loss.item()

    def update_ewc(self, X, y):
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).to(self.device)
        fisher = self._compute_fisher(X_t, y_t)
        optpar = {name: p.data.clone() for name, p in self.model.named_parameters()}
        self.fisher = fisher
        self.optpar = optpar

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            return self.model(X_t).cpu().numpy().squeeze()
