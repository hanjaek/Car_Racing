# car_train.py (학습 코드)

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# CarRacing 환경 생성 (v2 → v3 변경)
env = gym.make("CarRacing-v3", domain_randomize=False, render_mode="rgb_array")

# Monitor로 환경 감시 (로그 저장)
env = Monitor(env)

# 벡터 환경으로 래핑 (SAC 학습 안정성을 위해 필요)
env = DummyVecEnv([lambda: env])

# 학습된 모델이 있으면 불러와서 추가 학습, 없으면 새 모델 학습
try:
    model = SAC.load("sac_car_racing_best", env=env)
    print("✅ 기존 모델을 불러와서 추가 학습합니다.")
except:
    print("🚀 기존 모델이 없어서 새로 학습을 시작합니다.")
    model = SAC(
        "CnnPolicy",  # CNN 기반 정책 사용
        env,
        learning_rate=3e-4,  # 학습률
        buffer_size=100000,  # 리플레이 버퍼 크기
        batch_size=64,  # 배치 크기
        tau=0.005,  # 목표 네트워크 업데이트 속도
        gamma=0.99,  # 할인율
        train_freq=4,  # 4 스텝마다 학습
        gradient_steps=2,  # 업데이트 스텝
        verbose=1
    )

# 학습 수행 (최소 500만 스텝 권장)
model.learn(total_timesteps=10000000, log_interval=10)

# 모델 저장
model.save("sac_car_racing_best")
print("💾 학습이 완료되었습니다. 모델이 저장되었습니다.")
