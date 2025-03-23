import pandas as pd
import matplotlib.pyplot as plt

hil = pd.read_csv("tensorboard_logs/SAC_HIL_ceed1.csv.monitor.csv", skiprows=1)
no_hil = pd.read_csv("tensorboard_logs/SAC_ceed1.csv.monitor.csv", skiprows=1)

# 컬럼 확인해보기
print(hil.columns)  # ['r', 'l', 't']

# 그래프 그리기
plt.plot(hil['t'], hil['r'], label='HIL 개입', alpha=0.7)
plt.plot(no_hil['t'], no_hil['r'], label='일반 학습', alpha=0.7)

plt.xlabel("Time (s)")
plt.ylabel("Reward")
plt.title("HIL vs 일반 학습 보상 비교")
plt.legend()
plt.grid(True)
plt.show()
