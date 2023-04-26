import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.init as init
import numpy as np
from tqdm import tqdm

def cycloid_loss(target, output, theta, r):
    diff = abs(target-output)
    y = r * (1 - torch.cos(theta))
    return torch.mean(y*diff)

# def cycloid_loss(output, target, x):
#     a = 1  # adjust the amplitude as needed
#     b = 1  # adjust the period as needed
#     t = torch.linspace(0, 1 * torch.pi, len(x))  # use len(x) instead of len(output)
#     cycloid = a * (t - torch.sin(t) * b)
#     diff = torch.abs(output - target)  # calculate the absolute difference
#     cyc_factor = 0.5 + 0.5 * torch.cos((x - cycloid) / b)
#     loss = diff * cyc_factor  # weight the difference by cyc_factor
#     return loss

def mse_loss(output, target):
    return torch.mean((output - target) ** 2)

#parameter
num_epochs = 100

# Define the input data
# target = torch.linspace(1, 100, num_epochs).view(-1,1)
# x = torch.linspace(0, 1, num_epochs).view(-1,1)

x = torch.linspace(-2, 2, 100).view(-1,1)

# 방정식에 따른 y값 계산
target = 2 * x ** 3 - 4 * x ** 2 + 3 * x - 1

# Define the constant
theta_1 = torch.linspace(1 * np.pi, 2 * np.pi, num_epochs).view(-1,1)
theta_2 = torch.linspace(0, 1 * np.pi, num_epochs).view(-1,1)
r = 1

#Cycloid period
x_1 = r * (theta_1 - np.sin(theta_1))
x_2 = r * (theta_2 - np.sin(theta_2))
cycloid_graph_1 = -r * (1 - np.cos(theta_1))
cycloid_graph_2 = -r * (1 - np.cos(theta_2))

#위치 조정
cycloid_graph_1 = cycloid_graph_1
cycloid_graph_2_1 = 60*cycloid_graph_2+100
cycloid_graph_2_2 = cycloid_graph_2+100

derivatives = []
gradients = []
cycl_list = []
mse_list = []

# #cycloid list 생성
# for i in range(len(theta)):
#     # 미분값 계산
#     if i == 0:
#         deriv = (cycloid[i+1] - cycloid[i]) / (theta[i+1] - theta[i])
#     elif i == len(theta)-1:
#         deriv = (cycloid[i] - cycloid[i-1]) / (theta[i] - theta[i-1])
#     else:
#         deriv = (cycloid[i+1] - cycloid[i-1]) / (theta[i+1] - theta[i-1])
#     derivatives.append(-deriv.item())
#
#     # gradient 계산
#     theta_i = theta[i].requires_grad_()
#     cycloid_i = theta_i - torch.sin(theta_i)
#     cycloid_i.backward()
#     gradients.append(theta_i.grad.item())

# Define the model
model = torch.nn.Linear(1, 1)

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
cycl_losses = []
mse_losses = []
cycl_maps = []
mse_maps = []
cycl_weights_list = []
mse_weights_list = []
mse_bias_list = []

for i in tqdm(range(100)):  # repeat training for 10 times
    # init.xavier_normal_(model.weight)
    # # Train the model for 100 epochs with Cycloid Loss
    # losses = []
    # crr_weights_list = []
    # # for epoch in range(num_epochs):
    # #     # Zero the gradients
    # #     optimizer.zero_grad()
    # #     weights = model.weight.data
    # #     # Forward pass
    # #     output = model(x)
    # #     loss = cycloid_loss(output, target, x, r)
    # #     # loss = torch.t
    # #     #
    # #     # ensor(derivatives[epoch], requires_grad=True)
    # #
    # #     # Backward pass
    # #     loss.backward()
    # #     crr_weights_list.append(weights.item())
    # #     optimizer.step()
    # #
    # #     # Store the loss
    # #     losses.append(loss.item())
    # #
    # # # Store the losses
    # # cycl_losses.append(losses)
    # # cycl_maps.append(output.detach().numpy())
    # # cycl_weights_list.append(crr_weights_list)

    # Train the model for 100 epochs with MSE Loss
    init.xavier_normal_(model.weight)
    losses = []
    crr_weights_list = []
    crr_bias_list = []
    for epoch in range(num_epochs):
        # Zero the gradients
        optimizer.zero_grad()
        #weight 추적
        weights = model.weight.data
        bias = model.bias.data

        # Forward pass
        output = model(x)
        loss = mse_loss(output, target)

        # Backward pass
        loss.backward()
        crr_weights_list.append(weights.item())
        crr_bias_list.append(bias.item())
        optimizer.step()

        # Store the loss
        losses.append(loss.item())


    # Store the losses
    mse_losses.append(losses)
    mse_maps.append(output.detach().numpy())
    mse_weights_list.append(crr_weights_list)
    mse_bias_list.append(crr_bias_list)

# Compute the mean and variance of the losses and output maps
cycl_losses_mean = np.mean(cycl_losses, axis=0)
mse_losses_mean = np.mean(mse_losses, axis=0)
cycl_maps_mean = np.mean(cycl_maps, axis=0)
mse_maps_mean = np.mean(mse_maps, axis=0)
cycl_losses_var = np.var(cycl_losses, axis=0)
mse_losses_var = np.var(mse_losses, axis=0)
cycl_maps_var = np.var(cycl_maps, axis=0)
mse_maps_var = np.var(mse_maps, axis=0)
weight_mean = np.mean(mse_weights_list, axis=0)

##싸이클로이드 그래프와 가장 비슷한 cycl_losses 뽑기
# mse_losses_gradients = np.gradient(weight_mean, axis=0)
cycl_graph_gradients_1 = np.gradient(cycloid_graph_1, axis=0)
cycl_graph_gradients_2 = np.gradient(cycloid_graph_2, axis=0)
cycl_graph_gradients_2_1 = np.gradient(cycloid_graph_2_1, axis=0)
cycl_graph_gradients_2_2 = np.gradient(cycloid_graph_2_2, axis=0)

gradients_list = []
for i in range(len(mse_weights_list)):
    mse_losses_gradients = np.gradient(mse_weights_list[i], axis=0)
    gradients_diff_1 = abs(mse_losses_gradients - cycl_graph_gradients_1)
    gradients_diff_2 = abs(mse_losses_gradients - cycl_graph_gradients_2)
    gradients_diff_mean_1 = np.mean(gradients_diff_1, axis=0)
    gradients_diff_mean_2 = np.mean(gradients_diff_2, axis=0)
    if np.mean(gradients_diff_mean_1) >= np.mean(gradients_diff_mean_2):
        gradients_diff_mean = gradients_diff_mean_2
    else:
        gradients_diff_mean = gradients_diff_mean_1
    gradients_list.append(gradients_diff_mean)

min_mean_idx = np.argmin(gradients_list)
cycl_gradients_diff_mean_min = mse_losses[min_mean_idx]

# Plot the losses
# plt.plot(mse_maps, cycl_losses_mean, label="Cycloid Loss")
# # plt.fill_between(np.arange(num_epochs), max(cycl_losses), min(cycl_losses), alpha=0.2)
plt.plot(mse_losses_mean, label="MSE Loss")
# # plt.fill_between(np.arange(num_epochs), mse_losses_mean-mse_losses_var, mse_losses_mean+mse_losses_var, alpha=0.2)

plt.plot(cycl_gradients_diff_mean_min, label="Cycloid Loss Min")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# plt.plot(np.array(mse_weights_list[0]), mse_maps[0], label='mse & weight')
# plt.ylabel("mse_Maps")
# plt.legend()
# plt.show()

plt.plot(mse_weights_list[min_mean_idx], mse_losses[min_mean_idx], label='Cycloid Loss')
plt.plot(weight_mean, mse_losses_mean, label='MSE Losses Mean')
# plt.plot(x_1, cycloid_graph_1, label="Cycloid 1")
plt.plot(x_2, cycloid_graph_2, label="Cycloid 2")
plt.xlabel("Weight")
plt.ylabel("Loss")
plt.legend()
plt.show()