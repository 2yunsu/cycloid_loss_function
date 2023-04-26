import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.init as init

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

# Define the model
model = torch.nn.Linear(1, 1)

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)

# Define the input data
x = torch.linspace(0, 2 * torch.pi, 100).view(-1, 1)
target = torch.sin(x)

# Train the model with Cycloid Loss
num_epochs = 1000
cyc_losses = []

# Initialize the model parameters
init.xavier_normal_(model.weight)

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
    cyc_losses.append(loss.item())

    # Print the loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Cycloid Loss: {loss.item():.4f}")

# Train the model with MSE Loss
mse_losses = []

# Initialize the model parameters
init.xavier_normal_(model.weight)

for epoch in range(num_epochs):
    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    output = model(x)
    loss = torch.mean((output - target) ** 2)

    # Backward pass
    loss.backward()
    optimizer.step()

    # Store the loss
    mse_losses.append(loss.item())

    # Print the loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, MSE Loss: {loss.item():.4f}")

# Plot the losses
plt.plot(cyc_losses, label="Cycloid Loss")
plt.plot(mse_losses, label="MSE Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# # Plot the output and target with Cycloid Loss
# plt.plot(x.detach().numpy(), output.detach().numpy(), label="Cycloid Loss")
# plt.plot(x.detach().numpy(), target.detach().numpy(), label="Target")
# plt.title("Cycloid Loss")
# plt.legend()
# plt.show()
#
# # Plot the output and target with MSE Loss
# plt.plot(x.detach().numpy(), output.detach().numpy(), label="MSE Loss")
# plt.plot(x.detach().numpy(), target.detach().numpy(), label="Target")
# plt.title("MSE Loss")
# plt.legend()
# plt.show()
