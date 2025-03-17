import os
import numpy as np
import tensorflow as tf
from stable_baselines3 import SAC
from torch.utils.tensorboard import SummaryWriter

# 모델 저장된 폴더 경로
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "sac_car_racing_best.zip")

# TensorBoard 로그 저장 경로
LOG_DIR = "tensorboard_logs"
os.makedirs(LOG_DIR, exist_ok=True)
writer = SummaryWriter(LOG_DIR)

# 저장된 모델 불러오기
model = SAC.load(MODEL_PATH)

# 학습 과정에서 저장된 데이터 추출 (replay buffer 사용)
if model.replay_buffer is not None:
    for i in range(len(model.replay_buffer.rewards)):
        total_reward = np.sum(model.replay_buffer.rewards[i])  # 에피소드별 총 보상
        writer.add_scalar("Rewards/Total Reward per Episode", total_reward, i)

        # 손실 값 기록 (존재하는 경우)
        if hasattr(model.replay_buffer, 'losses'):
            loss = model.replay_buffer.losses[i] if model.replay_buffer.losses[i] is not None else 0
            writer.add_scalar("Loss/Training Loss", loss, i)

# TensorBoard 기록 저장
writer.close()

print(f"TensorBoard 로그가 '{LOG_DIR}' 폴더에 저장되었습니다. 실행하려면 다음 명령어를 사용하세요:")
print(f"tensorboard --logdir={LOG_DIR}")
