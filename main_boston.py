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
import pdb

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
num_epochs = 60
num_iter = 100
num_extract = 50
converging_threshold = 0.015

# Theta of Cycloid
theta_1 = torch.linspace(1 * np.pi, 2 * np.pi, num_epochs).view(-1,1)
theta_2 = torch.linspace(0, 1 * np.pi, num_epochs).view(-1,1)

# Train the model
mse_losses = []
mse_weights_list = []
converged_epochs = []
velocities_list = []

for i in tqdm(range(num_iter)):
    # Define the model
    model = nn.Linear(13, 1)
    model.bias.data.fill_(0)
    init.xavier_normal_(model.weight)
    # Define the optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    losses = [] #Gradient를 구하기 위함
    crr_weights_list = []
    velocities = []
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
        # Compare the weights before and after backward pass
        weight_before_backward = model.weight.data.clone()
        optimizer.step()
        weight_after_backward = model.weight.data.clone()
        weight_diff_backward = weight_after_backward - weight_before_backward #for measure velocity
        
        # Store the loss
        crr_weights_list.append(weights.clone())
        losses.append(loss.item())
        weight_diff_backward = torch.mean(torch.abs(weight_diff_backward))
        velocities.append(weight_diff_backward.item())


        # Check for convergence
        if len(converged_epochs) == i:
            if len(losses) > 1:
                # gradient = losses[-1] - losses[-2] # 현재 epoch과 이전 epoch의 loss 값 차이
                if abs(weight_diff_backward) < converging_threshold: # 기울기가 0에 가까워질 때마다 converged_epochs 리스트에 epoch 값을 추가
                    converged_epochs.append(epoch)
                if epoch == num_epochs - 1: # 한 epoch 끝까지 수렴 안 하면 마지막 값을 수렴점으로 지정
                    converged_epochs.append(epoch)

    # Store the losses
    mse_losses.append(losses)
    mse_weights_list.append(crr_weights_list)
    velocities_list.append(velocities)

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

    #위치 조정
    x_1 = (x_1 / x_2[-1]) * (crr_weights_tensor[-1] - crr_weights_tensor[0]) + crr_weights_tensor[0] #x축 맞추기
    x_2 = (x_2 / x_2[-1]) * (crr_weights_tensor[-1] - crr_weights_tensor[0]) + crr_weights_tensor[0]
    cycloid_graph_1_1 = cycloid_graph_1 + mse_losses[i][0] #y축 왼쪽 시작점 맞추기
    cycloid_graph_2_1 = cycloid_graph_2 + mse_losses[i][0]

    #모델 입력 뉴런 갯수만큼 확장
    cycloid_graph_1_1_expanded = cycloid_graph_1_1.expand(num_epochs, model.in_features)
    cycloid_graph_2_1_expanded = cycloid_graph_2_1.expand(num_epochs, model.in_features)
    #mse_losses_expanded = mse_losses_tensor.unsqueeze(1).expand(-1, model.in_features)

    #입력 갯수만큼의 싸이클로이드 곡선 생성
    cycl_graph_gradients_1_list = []
    cycl_graph_gradients_2_list = []
    #Gradient
    for j in range(model.in_features):
        cycl_graph_gradients_1 = np.gradient(cycloid_graph_1_1, x_1[:, j], axis=0)
        cycl_graph_gradients_2 = np.gradient(cycloid_graph_2_1, x_2[:, j], axis=0)
        cycl_graph_gradients_1_list.append(cycl_graph_gradients_1)
        cycl_graph_gradients_2_list.append(cycl_graph_gradients_2)

    #입력 갯수만큼의 weight 곡선 생성
    mse_weight_gradients_list = []
    for j in range(model.in_features):
        first_elements = [tensor[0, j] for tensor in mse_weights_list[i]]
        mse_weight_gradients = np.gradient(mse_losses[i], first_elements, axis=0) #i번째 Loss 함수의 gradient
        mse_weight_gradients_list.append(mse_weight_gradients)

    # Gradient 차이 계산
    gradients_diff_1_list = []
    gradients_diff_2_list = []
    for j in range(model.in_features):
        gradients_diff_1 = [x - y for x, y in zip(mse_weight_gradients_list[j], cycl_graph_gradients_1_list[j])]
        gradients_diff_2 = [x - y for x, y in zip(mse_weight_gradients_list[j], cycl_graph_gradients_2_list[j])]
        gradients_diff_1_list.append(gradients_diff_1)
        gradients_diff_2_list.append(gradients_diff_2)

    #두 리스트에서 더 작은 원소를 추출하여 새로운 리스트 형성
    #왜냐하면, 두 그래프의 차이가 크면 반대편의 Gradient를 계산하고 있는 것이기 때문
    gd_result = [np.minimum(np.abs(row1), np.abs(row2)) for row1, row2 in zip(gradients_diff_1_list, gradients_diff_2_list)]
    mean_gd_result = np.mean(gd_result)
    gradients_list.append(mean_gd_result)

    # #사이클로이드 그래프와 weight 그래프 렌더링. 주석 해제하고 디버그 모드로 실행할 것
    # for j in range(model.in_features):
    #     first_elements = [tensor[0, j] for tensor in mse_weights_list[i]]
    #     plt.plot(x_1[:, j], cycloid_graph_1_1, label="Cycloid", color="Orange")
    #     plt.plot(x_2[:, j], cycloid_graph_2_1, color="Orange")
    #     plt.plot(first_elements, mse_losses[i], label="MSE Loss Function")
    #     plt.xlabel("weight")
    #     plt.ylabel("Loss")
    #     plt.legend()
    #     # plt.show()
    #     plt.savefig('fig1.png')
    #     print()

# min_idx = np.argmin(gradients_list) #가장 비슷한 1개만 추출
# mse_losses_mean_min = mse_losses[min_idx]
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
print("converging threshold: ", converging_threshold)
print("Top 5% Cycloid Loss function Converge Mean at: ", np.mean(converged_10_list))
print("Top 5% low initial loss Mean: ", np.mean(init_converged_10_list))
print("Converge Mean: ", np.mean(converged_epochs))
print("Top 5% Cycloid Loss std: ", np.std(converged_10_list))
print("Top 5% low initial Loss std: ", np.std(init_converged_10_list))
print("MSE Loss std: ", np.std(converged_epochs))

# Plot the losses

# plt.plot(mse_losses_mean_min, label="Cycloid Loss")
plt.plot(mse_losses[0], label='All of MSE', color='tab:gray')
plt.plot(mse_losses[min_idx_10[0]], label='Top 5% MSE Similar to Cycloid', color='tab:cyan')
plt.plot(mse_losses[min_idx_10[0]], label='Low Initial Error', color='tab:orange')
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
plt.title("Boston Regression Loss per Epoch")
plt.savefig("/root/cycloid_loss_function/graph/boston_regression_loss_per_epoch.png")
plt.close()
# plt.show()
plt.clf()

pdb.set_trace()
for i in range(len(mse_losses)):
    plt.plot(converged_epochs[i], gradients_list[i], 'o', color='tab:gray')

plt.xlabel("gradient")
plt.ylabel("converged epoch")
plt.title("Boston Regression Loss per Converge")
plt.savefig("/root/cycloid_loss_function/graph/boston_regression_loss_per_converge.png")
plt.close()