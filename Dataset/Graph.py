import torch
from torch_geometric.data import Data
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


# Parameters
num_nodes = 100
num_node_features = 16
num_classes = 3
num_edges = 300  # roughly 3 edges per node


# Generate random node features
x = torch.randn((num_nodes, num_node_features), dtype=torch.float)


# Generate random edges (make sure source ≠ target)
edges = set()
while len(edges) < num_edges:
   src = np.random.randint(0, num_nodes)
   dst = np.random.randint(0, num_nodes)
   if src != dst:
       edges.add((src, dst))


edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()


# Generate random labels (for node classification)
y = torch.randint(0, num_classes, (num_nodes,), dtype=torch.long)


# Optional: Train/val/test masks (split nodes)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)


train_mask[:60] = True  # first 60 nodes for training
val_mask[60:80] = True  # next 20 for validation
test_mask[80:] = True   # last 20 for testing


# Create PyTorch Geometric Data object
data = Data(x=x, edge_index=edge_index, y=y,
           train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)


print(data)


# Optional: Visualize the graph using NetworkX
G = nx.Graph()
G.add_nodes_from(range(num_nodes))  # Ensure all nodes are added
G.add_edges_from(edge_index.t().tolist())


pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, node_color=y.tolist(), with_labels=True, cmap=plt.cm.Set1)
plt.title("Synthetic GNN Graph with Node Labels")
plt.show()
