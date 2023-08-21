import torch
import torch.nn.init as init
from torch import nn, optim
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import pandas as pd
import warnings

#에러 무시
warnings.filterwarnings("ignore")

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
dataset = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

#데이터셋 불러오기
# dataset = datasets.load_boston()
X, y = dataset, target

#텐서 자료구조로 변경
X = torch.FloatTensor(X)
y = torch.FloatTensor(y).unsqueeze(-1)

# normalize
X = (X - torch.mean(X)) / torch.std(X)

# criterion
criterion = nn.MSELoss()

def mse_loss(output, target):
    return torch.mean((output - target) ** 2)

#random seed
random_seed = 0
torch.manual_seed(random_seed)  # torch
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
np.random.seed(random_seed)  # numpy
random.seed(random_seed)  # random

#parameter
num_epochs = 80
num_iter = 1000
num_extract = 20

# Theta of Cycloid
theta_1 = torch.linspace(1 * np.pi, 2 * np.pi, num_epochs//2).view(-1,1)
theta_2 = torch.linspace(0, 1 * np.pi, num_epochs//2).view(-1,1)

# Train the model
mse_losses = []
mse_weights_list = []
converged_epochs = []

for i in tqdm(range(num_iter)):
    # Define the model
    model = nn.Linear(13, 1)
    model.bias.data.fill_(0)
    init.xavier_normal_(model.weight)
    # Define the optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    losses = [] #Gradient를 구하기 위함
    crr_weights_list = []
    crr_bias_list = []
    for epoch in range(num_epochs):
        # Zero the gradients
        optimizer.zero_grad()
        #weight 추적
        weights = model.weight.data
        bias = model.bias.data

        # Forward pass
        y_predicted = model(X)
        loss = criterion(y_predicted, y)

        # Backward pass
        loss.backward()
        crr_weights_list.append(weights.clone())
        # crr_bias_list.append(bias.item())
        optimizer.step()

        losses.append(loss.item())

        # # Loss 값을 저장하되 Loss가 이전 epoch보다 올라가면 저장하지 않음(Cycloid와 비교하기 위함)(실패)
        # if len(losses) == 0:
        #     losses.append(loss.item())
        # elif loss.item() < losses[-1]:
        #     losses.append(loss.item())
        # elif loss.item() >= losses[-1]:
        #     losses.append(losses[epoch-1])

        # Check for convergence
        if len(converged_epochs) == i:
            if len(losses) > 1:
                gradient = losses[-1] - losses[-2] # 현재 epoch과 이전 epoch의 loss 값 차이
                if abs(gradient) < 0.05: # 기울기가 0에 가까워질 때마다 converged_epochs 리스트에 epoch 값을 추가
                    converged_epochs.append(epoch)
                if epoch == num_epochs - 1: # 한 epoch 끝까지 수렴 안 하면 마지막 값을 수렴점으로 지정
                    converged_epochs.append(epoch)

    # Store the losses
    mse_losses.append(losses)
    mse_weights_list.append(crr_weights_list)

gradients_list = []
for i in tqdm(range(num_iter)):
    # Convert lists to tensors
    crr_weights_tensor = torch.stack(mse_weights_list[i])
    mse_losses_tensor = torch.tensor(mse_losses[i])

    # Cycloid 초기값
    r = abs(mse_losses_tensor[-1] - mse_losses_tensor[0])/2  # Cycloid 그래프의 높이 맞추기
    x_1 = r * (theta_1 - np.sin(theta_1))
    x_2 = r * (theta_2 - np.sin(theta_2))
    cycloid_graph_1 = -r * (1 - np.cos(theta_1))
    cycloid_graph_2 = -r * (1 - np.cos(theta_2))

    combine_cycl = torch.cat((cycloid_graph_1, cycloid_graph_2), dim=0)
    combine_cycl_np = combine_cycl.numpy()
    combine_cycl_list = combine_cycl_np.tolist()

    #위치 조정
    x_1 = (x_1 / x_2[-1]) * (crr_weights_tensor[-1] - crr_weights_tensor[0]) + crr_weights_tensor[0] #x축 맞추기
    x_2 = (x_2 / x_2[-1]) * (crr_weights_tensor[-1] - crr_weights_tensor[0]) + crr_weights_tensor[0]
    cycloid_graph_1_1 = combine_cycl + mse_losses[i][0] #y축 왼쪽 시작점 맞추기
    combine_x = torch.cat((x_1, x_2), dim=0)


    #모델 입력 뉴런 갯수만큼 확장
    cycloid_graph_1_1_expanded = cycloid_graph_1_1.expand(num_epochs, model.in_features)
    mse_losses_expanded = mse_losses_tensor.unsqueeze(1).expand(-1, model.in_features)

    #입력 갯수만큼의 싸이클로이드 곡선 생성
    cycl_graph_gradients_1_list = []
    #Gradient
    for j in range(model.in_features):
        cycl_graph_gradients_1 = np.gradient(cycloid_graph_1_1, combine_x[:, j], axis=0)
        cycl_graph_gradients_1_list.append(cycl_graph_gradients_1)

    #입력 갯수만큼의 weight 곡선 생성
    mse_weight_gradients_list = []
    for j in range(model.in_features):
        first_elements = [tensor[0, j] for tensor in mse_weights_list[i]]
        mse_weight_gradients = np.gradient(mse_losses[i], first_elements, axis=0) #i번째 Loss 함수의 gradient
        mse_weight_gradients_list.append(mse_weight_gradients)

    # Gradient 차이 계산
    gradients_diff_1_list = []
    for j in range(model.in_features):
        gradients_diff_1 = [x - y for x, y in zip(mse_weight_gradients_list[j], combine_cycl_np)]
        gradients_diff_1_list.append(gradients_diff_1)

    #두 리스트에서 더 작은 원소를 추출하여 새로운 리스트 형성
    mean_gd_result = np.mean(gradients_diff_1_list)
    gradients_list.append(mean_gd_result)

    #사이클로이드 그래프와 weight 그래프 렌더링. 주석 해제하고 디버그 모드로 실행할 것
    # for j in range(model.in_features):
    #     first_elements = [tensor[0, j] for tensor in mse_weights_list[i]]
    #     plt.plot(x_1[:, j], cycloid_graph_1_1, label="Cycloid", color="Orange")
    #     plt.plot(x_2[:, j], cycloid_graph_2_1, color="Orange")
    #     plt.plot(first_elements, mse_losses[i], label="MSE Loss Function")
    #     plt.xlabel("weight")
    #     plt.ylabel("Loss")
    #     plt.legend()
    #     plt.show()
    #     # plt.savefig('fig1.png')
    #     print()

min_idx = np.argmin(gradients_list)
mse_losses_mean_min = mse_losses[min_idx]
mse_losses_mean = np.mean(mse_losses, axis=0)

min_idx_10 = np.argsort(gradients_list)[:num_extract] #상위 n개 추출하여 비교

sorted_indices = sorted(range(len(mse_losses)), key=lambda i: mse_losses[i][0]) # 0번째 원소를 기준으로 리스트를 정렬한 후 인덱스를 정렬
init_loss_min_idx_10 = sorted_indices[:num_extract]  # 상위 n개 인덱스 추출

converged_10_list = []
for i in range(len(min_idx_10)):
    converged_10_list.append(converged_epochs[min_idx_10[i]])

init_converged_10_list = []
for i in range(len(init_loss_min_idx_10)):
    init_converged_10_list.append(converged_epochs[init_loss_min_idx_10[i]])

#when converge
# print("Top Cycloid Loss function Converge at: ", converged_epochs[min_idx])
print("Top 5% Cycloid Loss function Converge Mean at: ", np.mean(converged_10_list))
print("Top 5% low initial loss Mean: ", np.mean(init_converged_10_list))
print("Converge Mean: ", np.mean(converged_epochs))
print("Top 5% Cycloid Loss Varience: ", np.var(converged_10_list))
print("Top 5% low initial Loss Varience: ", np.var(init_converged_10_list))
print("MSE Loss Varience: ", np.var(converged_epochs))

# Plot the losses

# plt.plot(mse_losses_mean_min, label="Cycloid Loss")
plt.plot(mse_losses[0], label='All of MSE', color='tab:gray')
plt.plot(mse_losses[min_idx_10[0]], label='Top 5% MSE Similar to Cycloid', color='tab:cyan')
for i in range(len(mse_losses)):
    plt.plot(mse_losses[i], color='tab:gray')

for i in range(len(converged_10_list)):
    plt.plot(mse_losses[min_idx_10[i]], color='tab:cyan')

for i in range(len(init_converged_10_list)):
    plt.plot(mse_losses[init_loss_min_idx_10[i]], color='tab:orange')

plt.plot(mse_losses_mean, label="Mean of MSE", color='red')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
# plt.savefig("fig2.png")
plt.show()