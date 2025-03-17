import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# 모델 저장할 폴더 경로 설정
MODEL_DIR = "basic_model"
MODEL_PATH = os.path.join(MODEL_DIR, "sac_car_racing_best")

# 모델 폴더가 없으면 생성
os.makedirs(MODEL_DIR, exist_ok=True)

# CarRacing 환경 생성
env = gym.make("CarRacing-v3", domain_randomize=False, render_mode="rgb_array")

# Monitor로 환경 감시 (로그 저장)
env = Monitor(env)

# 벡터 환경으로 래핑 (SAC 학습 안정성을 위해 필요)
env = DummyVecEnv([lambda: env])

# 학습된 모델이 있으면 불러와서 추가 학습, 없으면 새 모델 학습
try:
    model = SAC.load(MODEL_PATH, env=env)
    print(f"✅ 기존 모델을 불러와서 추가 학습합니다. ({MODEL_PATH})")
except:
    print("🚀 기존 모델이 없어서 새로 학습을 시작합니다.")
    model = SAC(
        "CnnPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1000000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        verbose=1
    )

# 학습 수행 (100만 스텝으로 변경)
model.learn(total_timesteps=1000000)

# 모델 저장 (model 폴더 안에 저장됨)
model.save(MODEL_PATH)
print(f"💾 학습이 완료되었습니다. 모델이 저장되었습니다. ({MODEL_PATH})")
