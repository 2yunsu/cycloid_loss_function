import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.init as init
import numpy as np
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler

# Define the cycloid loss function
def cycloid_loss(output, target, x):
    a = 1  # adjust the amplitude as needed
    b = 1  # adjust the period as needed
    t = torch.linspace(0, 2 * torch.pi, len(x))  # use len(x) instead of len(output)
    cycloid = a * (t - torch.sin(t) * b)
    loss = torch.mean(torch.abs(output - target))
    cyc_factor = 0.5 + 0.5 * torch.cos((x - cycloid) / b)
    loss = cyc_factor * loss
    return loss.mean()

def mse_loss(output, target):
    return torch.mean(torch.pow(output - target, 2))

# Define the model
model = torch.nn.Linear(1, 1)

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Define the input data
x = torch.linspace(0, 2 * torch.pi, 100).view(-1, 1)
target = torch.linspace(1, 10, 100).view(-1, 1)
theta = np.linspace(0, 4 * np.pi, 100)
r = 1
x_1 = r * (theta - np.sin(theta))

# Train the model
num_epochs = 100
cycl_losses = []
mse_losses = []
cycl_maps = []
mse_maps = []

for i in tqdm(range(10)):  # repeat training for 10 times
    # Initialize the model parameters
    init.xavier_normal_(model.weight)

    # Train the model for 100 epochs with Cycloid Loss
    losses = []
    for epoch in range(num_epochs):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(x)
        loss = cycloid_loss(output, target, x)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Store the loss
        losses.append(loss.item())

    # Store the losses
    cycl_losses.append(losses)
    cycl_maps.append(output.detach().numpy())

    # Train the model for 100 epochs with MSE Loss
    losses = []
    for epoch in range(num_epochs):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(x)
        loss = mse_loss(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Store the loss
        losses.append(loss.item())

    # Store the losses
    mse_losses.append(losses)
    mse_maps.append(output.detach().numpy())

# Compute the mean and variance of the losses and output maps
cycl_losses_mean = np.mean(cycl_losses, axis=0)
mse_losses_mean = np.mean(mse_losses, axis=0)
cycl_maps_mean = np.mean(cycl_maps, axis=0)
mse_maps_mean = np.mean(mse_maps, axis=0)
cycl_losses_var = np.var(cycl_losses, axis=0)
mse_losses_var = np.var(mse_losses, axis=0)
cycl_maps_var = np.var(cycl_maps, axis=0)
mse_maps_var = np.var(mse_maps, axis=0)

# Plot the losses
plt.plot(cycl_losses_mean, label="Cycloid Loss")
# plt.fill_between(np.arange(num_epochs), max(cycl_losses), min(cycl_losses), alpha=0.2)
plt.plot(mse_losses_mean, label="MSE Loss")
# plt.fill_between(np.arange(num_epochs), mse_losses_mean-mse_losses_var, mse_losses_mean+mse_losses_var, alpha=0.2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Calculate the mean and variance of the cycloid
cycl_mean = torch.mean(output).item()
cycl_var = torch.var(output).item()

# Print the mean and variance of the cycloid
print(f"Mean of the cycloid: {cycl_mean:.4f}")
print(f"Variance of the cycloid: {cycl_var:.4f}")

