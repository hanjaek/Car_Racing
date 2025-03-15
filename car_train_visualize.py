import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

# 모델 저장된 폴더 경로
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "sac_car_racing_best.zip")

# 저장된 모델 불러오기
model = SAC.load(MODEL_PATH)

# 학습 과정에서 저장된 데이터 추출 (replay buffer 사용)
rewards = []
losses = []

if model.replay_buffer is not None:
    for i in range(len(model.replay_buffer.rewards)):
        rewards.append(np.sum(model.replay_buffer.rewards[i]))  # 에피소드별 총 보상
        losses.append(model.replay_buffer.losses[i] if hasattr(model.replay_buffer, 'losses') else None)

# 보상 변화 그래프
plt.figure(figsize=(10, 5))
plt.plot(rewards, label="Total Reward per Episode", color='blue')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Reward Progress")
plt.legend()
plt.grid()
plt.show()

# 손실 값 그래프 (만약 존재하면)
if any(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Loss per Update", color='red')
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Progress")
    plt.legend()
    plt.grid()
    plt.show()
