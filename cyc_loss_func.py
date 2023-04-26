import torch
import matplotlib.pyplot as plt

def cycloid_loss(output, target, x, i):
    a = 1  # adjust the amplitude as needed
    b = 1  # adjust the period as needed
    t = torch.linspace(0, 2 * torch.pi, len(x))  # use len(x) instead of len(output)
    cycloid = a * (t - torch.sin(t) * b)
    diff = torch.abs(output - target)  # calculate the absolute difference
    cyc_factor = 0.5 + 0.5 * torch.cos((x - cycloid) / b)
    loss = diff * cyc_factor[i]  # weight the difference by cyc_factor
    return cycloid[i]


x = torch.linspace(0, 2 * torch.pi, 100)
output = torch.linspace(1, 1, 100)
target = torch.linspace(1, 10, 100)

cycl_losses = []
for i in range(len(x)):
    cycl_losses.append(cycloid_loss(output[i], target[i], x, i).item())

L1_losses = []
for i in range(len(x)):
    L1_losses.append(abs((output[i]-target[i]).item()))

MSE_losses = []
for i in range(len(x)):
    MSE_losses.append(abs((output[i]-target[i]).item())**2)

plt.plot(cycl_losses, label="Cycloid Loss")
plt.plot(L1_losses, label="L1 Loss")
plt.plot(MSE_losses, label="MSE_loss")
plt.axis('equal')
plt.legend()
plt.show()