# car_trained.py (학습된 모델 실행 코드)

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# CarRacing 환경 생성 (v2 → v3로 변경)
env = gym.make("CarRacing-v3", domain_randomize=False, render_mode="human")

# Monitor로 환경 감시 (로그 저장)
env = Monitor(env)

# 벡터 환경으로 래핑 (테스트 환경에서는 없어도 가능)
env = DummyVecEnv([lambda: env])

# 학습된 모델 로드
model = SAC.load("sac_car_racing_best")
print("✅ 학습된 모델을 불러왔습니다. 자율주행을 시작합니다!")

obs, _ = env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)  # 학습된 정책 사용
    obs, reward, terminated, truncated, _ = env.step(action)  # v3에서는 terminated, truncated 반환
    done = terminated or truncated  # 종료 조건 업데이트
    env.render()
