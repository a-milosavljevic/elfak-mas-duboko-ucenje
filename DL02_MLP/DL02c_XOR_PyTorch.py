import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

print("PyTorch version:", torch.__version__)

# XOR dataset
data = np.array([
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
], dtype=np.float32)

inputs = torch.tensor(data[:, :2])
outputs = torch.tensor(data[:, 2:], dtype=torch.float32)

# Define model
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 5),
            nn.Sigmoid(),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = XORNet()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1.0, momentum=0.0)

# Training loop
losses = []
accuracies = []
for epoch in range(1000):
    optimizer.zero_grad()
    outputs_pred = model(inputs)
    loss = criterion(outputs_pred, outputs)
    loss.backward()
    optimizer.step()

    # Calculate accuracy
    predictions = (outputs_pred > 0.5).float()
    acc = (predictions == outputs).float().mean().item()

    losses.append(loss.item())
    accuracies.append(acc)

    if epoch == 0 or (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f} | Accuracy: {acc:.4f}")
        
# Plot
plt.plot(losses, label="Loss")
plt.plot(accuracies, label="Accuracy")
plt.legend()
plt.show()

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(inputs)
    loss = criterion(predictions, outputs)
    acc = ((predictions > 0.5).float() == outputs).float().mean().item()
    print(f"\nFinal evaluation -> Loss: {loss.item():.4f}, Accuracy: {acc:.4f}")

    for x, t, y in zip(inputs, outputs, predictions):
        print(x.numpy(), '->', t.item(), '=>', y.item())
