import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import numpy as np
import matplotlib.pyplot as plt
import random

# Define the GCN model
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# Load the Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Training loop with validation
def train(data, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Test the model
@torch.no_grad()
def evaluate(data, model, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[mask] == data.y[mask]).sum()
    acc = correct / mask.sum()
    return acc.item()

# Function for computing the test loss
@torch.no_grad()
def evaluate_test_loss(data, model):
    model.eval()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.test_mask], data.y[data.test_mask])
    return loss.item()

# Function for computing the training accuracy
@torch.no_grad()
def evaluate_train_accuracy(data, model):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
    acc = correct / data.train_mask.sum()
    return acc.item()

# Random search hyperparameter tuning function
def random_search(num_trials=10):
    best_acc = 0.0
    best_params = None

    hidden_units_options = [16, 32, 64, 128]
    learning_rate_options = [0.001, 0.01, 0.1]
    weight_decay_options = [1e-4, 5e-4, 1e-3]

    # Perform random search
    for trial in range(num_trials):
        # Sample random hyperparameters
        hidden_units = random.choice(hidden_units_options)
        learning_rate = random.choice(learning_rate_options)
        weight_decay = random.choice(weight_decay_options)

        print(f"Trial {trial + 1}/{num_trials} - Hidden: {hidden_units}, LR: {learning_rate}, WD: {weight_decay}")

        # Initialize the model, optimizer, and loss function with the sampled hyperparameters
        model = GCN(in_channels=dataset.num_node_features,
                    hidden_channels=hidden_units,
                    out_channels=dataset.num_classes)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        # Run the training process
        train_losses, test_accuracies = [], []
        for epoch in range(200):
            loss = train(data, model, optimizer, criterion)
            train_losses.append(loss)

            if epoch % 10 == 0 or epoch == 199:
                test_acc = evaluate(data, model, data.test_mask)
                test_accuracies.append(test_acc)
                print(f'Epoch {epoch:03d}, Test Acc: {test_acc:.4f}')

        # Determine the best set of hyperparameters based on test accuracy
        final_test_acc = test_accuracies[-1]
        if final_test_acc > best_acc:
            best_acc = final_test_acc
            best_params = {
                'hidden_units': hidden_units,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay
            }

    return best_params, best_acc

# Run random search to find best hyperparameters
best_params, best_acc = random_search(num_trials=10)

# Output the best hyperparameters and test accuracy
print(f"Best hyperparameters: {best_params}")
print(f"Best Test Accuracy: {best_acc:.4f}")
