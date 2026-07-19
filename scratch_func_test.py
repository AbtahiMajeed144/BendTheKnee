import torch
import torch.nn as nn
import torch.func

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)
    def forward(self, t, x):
        return self.fc(x) * t

net = SimpleNet()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

t = torch.tensor([0.5])
xt = torch.randn(1, 2, requires_grad=True)
epsilon = torch.randn(1, 2)

def func(x):
    return net(t, x)

vt, jvp_out = torch.func.jvp(func, (xt,), (epsilon,))
vt2, vjp_fn = torch.func.vjp(func, xt)
vjp_out = vjp_fn(epsilon)[0]

loss = torch.mean((jvp_out - vjp_out)**2)
loss.backward()

print("Loss:", loss.item())
for name, param in net.named_parameters():
    print(name, "grad:", param.grad)
