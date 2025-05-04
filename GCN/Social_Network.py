import torch
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx
import networkx as nx                                                           #Library for creating, manipulating, and studying the structure, dynamics, functions of complex networks.
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import Linear
from torch_geometric.nn import GCNConv


# Load the Karate Club dataset
dataset = KarateClub()
data = dataset[0]


print(dataset.num_features)


import pandas as pd
# Convert node features to a pandas DataFrame
node_features_df = pd.DataFrame(data.x.numpy(), columns=[f'feature_{i}' for i in range(data.x.shape[1])])
node_features_df['node'] = node_features_df.index
node_features_df.set_index('node', inplace=True)


# Convert edge indices to a pandas DataFrame
edge_index_df = pd.DataFrame(data.edge_index.numpy().T, columns=['source', 'target'])


# Convert labels to a pandas DataFrame
labels_df = pd.DataFrame(data.y.numpy(), columns=['label'])
labels_df['node'] = labels_df.index
labels_df.set_index('node', inplace=True)


print(node_features_df)


# Print the number of nodes
num_nodes = data.num_nodes
print(f"Number of nodes: {num_nodes}")


# Print the number of edges
num_edges = data.edge_index.shape[1]
print(f"Number of edges: {num_edges}")


# Print node features DataFrame
print(f"\nNode features (shape: {node_features_df.shape}):")
print(node_features_df.head())  # Print only the first few rows for brevity


# Print edge indices DataFrame
print(f"\nEdge indices (shape: {edge_index_df.shape}):")
print(edge_index_df.head())  # Print only the first few rows for brevity


# Print labels DataFrame
print(f"\nLabels (shape: {labels_df.shape}):")
print(labels_df.head())  # Print only the first few rows for brevity


# Number of countries
num_countries = 4


# Assign students to countries (labels)
np.random.seed(42)  # For reproducibility
countries = torch.tensor(np.random.choice(num_countries, data.num_nodes))


# Update the labels in the data object
data.y = countries


# Verify the data
print(data)
print(f'x = {data.x.shape}')
print(data.x)
print(f'edge_index = {data.edge_index.shape}')
print(data.edge_index)
print(f'y = {data.y.shape}')
print(data.y)


# Check if the graph is as expected
G = to_networkx(data, to_undirected=True)
plt.figure(figsize=(12,12))
plt.axis('off')
nx.draw_networkx(G,
               pos=nx.spring_layout(G, seed=0),
               with_labels=True,
               node_size=800,
               node_color=data.y,
               cmap="hsv",
               vmin=-2,
               vmax=3,
               width=0.8,
               edge_color="grey",
               font_size=14
               )
plt.show()


# Define the model
class GCN(torch.nn.Module):
   def __init__(self):
       super().__init__()
       self.gcn = GCNConv(dataset.num_features, 3)
       self.out = Linear(3, num_countries)
   def forward(self, x, edge_index):
       h = self.gcn(x, edge_index).relu()
       z = self.out(h)
       return h, z


model = GCN()
print(model)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)


# Calculate accuracy
def accuracy(pred_y, y):
   return (pred_y == y).sum() / len(y)


# Data for animations
embeddings = []
losses = []
accuracies = []
outputs = []


# Training loop
for epoch in range(200):
   optimizer.zero_grad()
   h, z = model(data.x, data.edge_index)
   loss = criterion(z, data.y)
   acc = accuracy(z.argmax(dim=1), data.y)
   loss.backward()
   optimizer.step()
   embeddings.append(h)
   losses.append(loss)
   accuracies.append(acc)
   outputs.append(z.argmax(dim=1))
   if epoch % 10 == 0:
       print(f'Epoch {epoch:>3} | Loss: {loss:.2f} | Acc: {acc*100:.2f}%')
from IPython.display import HTML
from matplotlib import animation


plt.rcParams["animation.bitrate"] = 3000


def animate(i):
   G = to_networkx(data, to_undirected=True)
   nx.draw_networkx(G,
                   pos=nx.spring_layout(G, seed=0),
                   with_labels=True,
                   node_size=800,
                   node_color=outputs[i].numpy(),
                   cmap="hsv",
                   vmin=-2,
                   vmax=3,
                   width=0.8,
                   edge_color="grey",
                   font_size=14
                   )
   plt.title(f'Epoch {i} | Loss: {losses[i].item():.2f} | Acc: {accuracies[i].item()*100:.2f}%',
             fontsize=18, pad=20)


fig = plt.figure(figsize=(12, 12))
plt.axis('off')


from mpl_toolkits.mplot3d import Axes3D


# Get final embeddings and predictions
final_h, final_z = model(data.x, data.edge_index)
predicted_labels = final_z.argmax(dim=1)


# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')


# Extract the 3D coordinates from the final embeddings
x = final_h[:, 0].detach().numpy()
y = final_h[:, 1].detach().numpy()
z = final_h[:, 2].detach().numpy()
colors = predicted_labels.detach().numpy()


# Plot the nodes
scatter = ax.scatter(x, y, z, c=colors, cmap='hsv', s=300)


# Annotate node IDs
for i in range(len(x)):
   ax.text(x[i], y[i], z[i], str(i), fontsize=9, color='black')


# Set axis labels and title
ax.set_title("3D Visualization of Node Embeddings", fontsize=15)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
