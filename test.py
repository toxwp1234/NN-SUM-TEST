import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class NewNet(nn.Module):
    def __init__(self):
        super(NewNet, self).__init__()
        self.Layer1 = nn.Linear(2, 8)
        self.Layer2 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.Layer1(x))
        x = self.Layer2(x)
        return x

siec = NewNet()
optimizer = torch.optim.Adam(siec.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

epochs = 50
ile_batchow = 100
seria = 32

for epoch in range(epochs):
    training_loss = 0.0
    siec.train()

    for batch in range(ile_batchow):
        for rep in range(seria):
            x1 = random.randint(0, 1000)
            x2 = random.randint(0, 1000)

            # Skalowanie danych wejściowych
            xx = torch.tensor([[x1 / 1000, x2 / 1000]], dtype=torch.float32)
            # Skalowanie targetu
            valid = torch.tensor([[ (x1 + x2) / 2000 ]], dtype=torch.float32)

            optimizer.zero_grad()
            output = siec(xx)
            loss = loss_fn(output, valid)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
    print(f"Epoch {epoch+1}, Training loss: {training_loss / (ile_batchow * seria):.6f}")


# TEST
print("\nTestowanie po treningu:\n")
siec.eval()
for _ in range(10):
    x1 = random.randint(0, 1000)
    x2 = random.randint(0, 1000)
    xx = torch.tensor([[x1 / 1000, x2 / 1000]], dtype=torch.float32)
    output = siec(xx).item() * 2000  # Przeskalowanie z powrotem
    print(f"{x1} + {x2} ≈ {output:.2f} (expected {x1 + x2})")
