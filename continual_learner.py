import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
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

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def add(self, x, y):
        self.buffer.append((x, y))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        X = torch.stack([b[0] for b in batch])
        y = torch.tensor([b[1] for b in batch], dtype=torch.float32)
        return X, y
    def __len__(self):
        return len(self.buffer)

def compute_fisher_information(model, dataloader, device):
    """Approximate Fisher information matrix for EWC."""
    model.eval()
    fisher = {}
    for name, param in model.named_parameters():
        fisher[name] = torch.zeros_like(param)
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        model.zero_grad()
        output = model(X).squeeze()
        loss = nn.MSELoss()(output, y)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.detach() ** 2
    for name in fisher:
        fisher[name] /= len(dataloader)
    model.train()
    return fisher

def train_step_replay(model, optimizer, criterion, X, y, replay_buffer, batch_size, device):
    # Train on current batch
    X, y = X.to(device), y.to(device)
    optimizer.zero_grad()
    pred = model(X).squeeze()
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    # Add to replay buffer
    for i in range(len(X)):
        replay_buffer.add(X[i].cpu(), y[i].cpu().item())
    # Replay step
    if len(replay_buffer) >= batch_size:
        X_rep, y_rep = replay_buffer.sample(batch_size)
        X_rep, y_rep = X_rep.to(device), y_rep.to(device)
        optimizer.zero_grad()
        pred_rep = model(X_rep).squeeze()
        loss_rep = criterion(pred_rep, y_rep)
        loss_rep.backward()
        optimizer.step()
    return loss.item()

def train_step_ewc(model, optimizer, criterion, X, y, ewc_lambda, fisher, old_params, device):
    X, y = X.to(device), y.to(device)
    optimizer.zero_grad()
    pred = model(X).squeeze()
    loss = criterion(pred, y)
    # EWC penalty
    ewc_loss = 0.0
    for name, param in model.named_parameters():
        if name in fisher:
            ewc_loss += (fisher[name] * (param - old_params[name]) ** 2).sum()
    total_loss = loss + ewc_lambda * ewc_loss
    total_loss.backward()
    optimizer.step()
    return loss.item()
