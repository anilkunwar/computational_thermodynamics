import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# Step 1: Data Preparation
# Node features: [x_Sn, x_Ag, x_Cu, T, phase_type]
node_features = torch.tensor([
    [0.8, 0.1, 0.1, 600, 0],  # LIQUID at composition [0.8, 0.1, 0.1], T=600K
    [0.8, 0.05, 0.15, 600, 0], # LIQUID at composition [0.8, 0.05, 0.15], T=600K
    [0.4, 0.3, 0.3, 500, 1],   # FCC_A1 at composition [0.4, 0.3, 0.3], T=500K
    [0.6, 0.2, 0.2, 500, 2],   # BCT_A5 at composition [0.6, 0.2, 0.2], T=500K
], dtype=torch.float)

# Enthalpy values (target) for each node
enthalpy = torch.tensor([100.0, 105.0, 150.0, 120.0], dtype=torch.float)

# Edge indices: connections between nodes (e.g., liquid to solid phases)
edge_index = torch.tensor([
    [0, 2], [2, 0],  # LIQUID [0] <-> FCC_A1 [2]
    [0, 3], [3, 0],  # LIQUID [0] <-> BCT_A5 [3]
    [1, 2], [2, 1],  # LIQUID [1] <-> FCC_A1 [2]
    [1, 3], [3, 1],  # LIQUID [1] <-> BCT_A5 [3]
], dtype=torch.long).t().contiguous()

# Edge features: [relationship_type, delta_x_Sn, delta_x_Ag, delta_x_Cu, delta_T]
edge_features = torch.tensor([
    [0, -0.4, 0.2, 0.2, 100],   # LIQUID [0] -> FCC_A1 [2]
    [0, 0.4, -0.2, -0.2, -100], # FCC_A1 [2] -> LIQUID [0]
    [0, -0.2, 0.1, 0.1, 100],   # LIQUID [0] -> BCT_A5 [3]
    [0, 0.2, -0.1, -0.1, -100], # BCT_A5 [3] -> LIQUID [0]
    [0, -0.4, 0.25, 0.15, 100], # LIQUID [1] -> FCC_A1 [2]
    [0, 0.4, -0.25, -0.15, -100],# FCC_A1 [2] -> LIQUID [1]
    [0, -0.2, 0.15, 0.05, 100], # LIQUID [1] -> BCT_A5 [3]
    [0, 0.2, -0.15, -0.05, -100],# BCT_A5 [3] -> LIQUID [1]
], dtype=torch.float)

# Create PyTorch Geometric Data object
data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, y=enthalpy)

# Step 2: Define the GNN Model
class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(5, 16)  # 5 input features -> 16 hidden
        self.conv2 = GCNConv(16, 1)  # 16 hidden -> 1 output (enthalpy)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Step 3: Training the Model
model = GNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.mse_loss(out.squeeze(), data.y)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Step 4: Inference
# Predict enthalpy for a new liquid node
new_node_features = torch.tensor([[0.75, 0.15, 0.1, 550, 0]], dtype=torch.float)  # LIQUID
new_edge_index = torch.tensor([[0, 2], [0, 3], [2, 0], [3, 0]], dtype=torch.long).t().contiguous()
new_edge_features = torch.tensor([
    [0, -0.35, 0.15, -0.2, 50],   # New LIQUID -> FCC_A1 [2]
    [0, -0.15, -0.05, -0.1, 50],  # New LIQUID -> BCT_A5 [3]
    [0, 0.35, -0.15, 0.2, -50],   # FCC_A1 [2] -> New LIQUID
    [0, 0.15, 0.05, 0.1, -50],    # BCT_A5 [3] -> New LIQUID
], dtype=torch.float)
new_data = Data(x=new_node_features, edge_index=new_edge_index, edge_attr=new_edge_features)

model.eval()
with torch.no_grad():
    predicted_enthalpy = model(new_data)
    print(f"\nPredicted Enthalpy for new liquid node: {predicted_enthalpy.item():.4f}")
