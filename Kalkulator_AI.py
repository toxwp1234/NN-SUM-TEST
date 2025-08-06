import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class NewNet(nn.Module):
    def __init__(self):
        super(NewNet, self).__init__()
        self.Layer1 = nn.Linear(2, 3)
        self.Layer2 = nn.Linear(3, 1)

    def forward(self, x):
        x = F.relu(self.Layer1(x))
        x = self.Layer2(x)
        return x

siec = NewNet()
optimizer = torch.optim.SGD(siec.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

epochs = 5
ile_batchow = 150
seria = 32

for epoch in range(epochs):
    training_loss = 0.0
    siec.train()

    for batch in range(ile_batchow):
        for rep in range(seria):
            x1 = random.randint(0, 1000)
            x2 = random.randint(0, 1000)


            skala = 1000

            x1 = x1/skala
            x2 = x2/skala

            xx = torch.tensor([[x1, x2]], dtype=torch.float32)
            valid = torch.tensor([[x1 + x2]], dtype=torch.float32)

            optimizer.zero_grad()
            output = siec(xx)
            loss = loss_fn(output, valid)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
    print(f"Epoch {epoch+1}, Training loss: {training_loss / (ile_batchow * seria):.4f}")



for x in range(50):

    x1 = random.randint(0, 1000)
    x2 = random.randint(0, 1000)
    
    c1=x1
    c2=x2

    skala = 1000

    x1 = x1/skala
    x2 = x2/skala

    xx = torch.tensor([[x1, x2]], dtype=torch.float32)


    print(f"{x1*skala} + {x2*skala} = {float(siec(xx).detach()[0])*skala} | Prawidłowe : {c1+c2}")




liczba1 = -1

while liczba1 != -9:

    x1 = int(input("1 Liczba"))
    x2 = int(input("2 Liczba"))
             
    
    c1=x1
    c2=x2

    skala = 1000

    x1 = x1/skala
    x2 = x2/skala

    xx = torch.tensor([[x1, x2]], dtype=torch.float32)


    print(f"{x1*skala} + {x2*skala} = {float(siec(xx).detach()[0])*skala} | Prawidłowe : {c1+c2}")