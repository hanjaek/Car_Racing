import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# 모델 저장된 폴더 경로
MODEL_DIR = "ceed_model_v0"
MODEL_PATH = os.path.join(MODEL_DIR, "sac_car_racing_best")

# ✅ SEED 설정 (항상 동일한 트랙 등장)
SEED = 1  

# CarRacing 환경 생성 (테스트 모드)
def make_env():
    def _init():
        env = gym.make("CarRacing-v3", domain_randomize=False, render_mode="human")
        env.reset(seed=SEED)  # ✅ 트랙 고정
        return env
    return _init

# ✅ 환경 생성 (SEED 적용)
env = DummyVecEnv([make_env()])
env.seed(SEED)

# 학습된 모델 로드
model = SAC.load(MODEL_PATH)
print(f"✅ 학습된 모델을 불러왔습니다. ({MODEL_PATH}) 자율주행을 시작합니다!")

# 환경 초기화
obs = env.reset()
env.render()  # 강제로 한 번 화면을 띄움

done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)  # 학습된 정책 사용
    obs, reward, done, info = env.step(action)  

    env.render()
