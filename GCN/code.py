import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import numpy as np
import matplotlib.pyplot as plt

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

# Initialize the model, optimizer, and loss function
model = GCN(
    in_channels=dataset.num_node_features,
    hidden_channels=16,
    out_channels=dataset.num_classes
)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

# Training loop with validation
def train(data, model):
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

# Run the training process and store the metrics
train_losses, train_accuracies, test_accuracies, test_losses = [], [], [], []

for epoch in range(200):
    # Train and record the training loss
    loss = train(data, model)
    train_losses.append(loss)

    # Evaluate on train set
    train_acc = evaluate_train_accuracy(data, model)
    train_accuracies.append(train_acc)

    # Evaluate on test set periodically
    if epoch % 10 == 0 or epoch == 199:
        test_acc = evaluate(data, model, data.test_mask)
        test_losses.append(evaluate_test_loss(data, model))
        test_accuracies.append(test_acc)
        print(f'Epoch {epoch:03d}, Train Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Test Loss: {test_losses[-1]:.4f}')

print("Training completed.")

# Plot all four metrics in separate subplots
plt.figure(figsize=(15, 12))

# Plot Training Loss
plt.subplot(2, 2, 1)
plt.plot(train_losses, label='Training Loss', color='blue')
plt.title("Training Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Plot Training Accuracy
plt.subplot(2, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy', color='orange')
plt.title("Training Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Plot Test Loss
plt.subplot(2, 2, 3)
plt.plot(test_losses, label='Test Loss', color='green')
plt.title("Test Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Plot Test Accuracy
plt.subplot(2, 2, 4)
plt.plot(test_accuracies, label='Test Accuracy', color='red')
plt.title("Test Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()

# Interactive Graph Visualization
import plotly.graph_objs as go
import networkx as nx

def plot_interactive_graph():
    edge_index = data.edge_index.cpu().numpy()
    G = nx.Graph()
    for i in range(edge_index.shape[1]):
        G.add_edge(edge_index[0, i], edge_index[1, i])

    # Node information
    labels = {i: data.y[i].item() for i in range(data.num_nodes)}
    colors = [data.y[i].item() for i in range(data.num_nodes)]

    # Graph layout
    pos = nx.spring_layout(G, k=0.5)

    # Prepare data for Plotly
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Node visualization
    node_x, node_y = [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=10,
            color=colors,
            colorbar=dict(
                thickness=15,
                title='Node Class',
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        ),
        text=[f"Node {i}, Class: {labels[i]}" for i in range(data.num_nodes)],
        hoverinfo='text'
    )

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="Interactive Visualization of Cora Dataset",
                        titlefont_size=16,
                        showlegend=False,
                        hovermode="closest",
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    ))
    fig.show()

# Visualize the graph
plot_interactive_graph()
