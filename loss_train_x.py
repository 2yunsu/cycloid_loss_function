import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.init as init
import numpy as np
from tqdm import tqdm

def cycloid(target, output, theta, r, i):
    diff = abs(target[i]-output[i])
    y = r * (1 - np.cos(theta))
    multi_result = y[i]*diff
    devide_result = multi_result / abs(target[i]-output[i])
    return devide_result

def mse_loss(output, target, i):
    return (output[i] - target[i])**2

# Define the input data
target = torch.linspace(2.414, 1, 100).tolist()
pred = torch.linspace(1, 1, 100).tolist()

# Define the constant
theta = np.linspace(0, 1 * np.pi, 100)
r = 1

#x의 범위
x = r * (theta - np.sin(theta))

cycl_list = []
mse_list = []

for i in range(len(x)):
    y = cycloid(target, pred, theta, r, i)
    cycl_list.append(-y+2)

for i in range(len(x)):
    y = mse_loss(target, pred, i)
    mse_list.append(y)

# Define the model
model = torch.nn.Linear(1, 1)

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
r = 1
x_1 = r * (theta - np.sin(theta))

# 시각화
# plt.plot(x[:250]-3.29875, -y[:250]+2)
plt.plot(x, cycl_list, label= "Cycloid Loss")
plt.plot(x, mse_list, label= "MSE Loss")
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()