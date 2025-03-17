import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# 모델 저장된 폴더 경로
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "sac_car_racing_best")

# CarRacing 환경 생성 (테스트 모드)
env = gym.make("CarRacing-v3", domain_randomize=False, render_mode="human")

# Monitor로 환경 감시
env = Monitor(env)

# 벡터 환경으로 래핑 (테스트 환경에서는 없어도 가능)
env = DummyVecEnv([lambda: env])

# 학습된 모델 로드
model = SAC.load(MODEL_PATH)
print(f"✅ 학습된 모델을 불러왔습니다. ({MODEL_PATH}) 자율주행을 시작합니다!")

# 환경 초기화
obs = env.reset()
env.render()  # 강제로 한 번 화면을 띄움

done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)  # 학습된 정책 사용

    # 🔥 수정된 부분: 반환값을 4개만 받도록 변경!
    obs, reward, done, info = env.step(action)  

    env.render()
