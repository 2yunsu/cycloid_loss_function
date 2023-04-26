import numpy as np
import matplotlib.pyplot as plt

# 입력 데이터 생성
x = [i for i in range(1, 101)]
y_true = x
y_pred = [0 for i in range(1, 101)]  # 예측값

def cycloid(theta, r):
    x = r * (theta - np.sin(theta))
    y = r * (1 - np.cos(theta))
    return x, y

theta = np.linspace(0, 4 * np.pi, 1000)
r = 1

x, y = cycloid(theta, r)

# 시각화
plt.plot(x[250:500]-3.29875, -y[250:500]+2)
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()