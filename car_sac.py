import gymnasium as gym
from stable_baselines3 import SAC
import os
import numpy as np

class CustomCarRacingEnv(gym.Wrapper):
    def __init__(self, env):
        super(CustomCarRacingEnv, self).__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 🎯 사용자 정의 보상 적용
        reward = self.custom_reward(obs, action)

        return obs, reward, terminated, truncated, info

    def custom_reward(self, obs, action):
        """
        도로 중심을 유지하고 정상적인 주행을 유도하는 보상 함수
        """
        speed_reward = obs[4]  # 속도 유지 보상 (obs[4]는 속도 정보)
        track_position = obs[1]  # 트랙 중심에서의 거리 (obs[1]은 도로 중심에서의 거리)
        angle = obs[3]  # 차량 방향과 도로 방향의 차이 (obs[3]은 방향 정보)

        # 도로 중심을 따라가도록 유도하는 보상
        track_reward = 1.0 - abs(track_position)

        # 도로를 벗어나면 패널티 부여
        off_road_penalty = -10 if abs(track_position) > 0.9 else 0

        # 차량이 도로를 따라가도록 유도
        angle_penalty = -abs(angle)

        # 총 보상 계산
        total_reward = speed_reward + track_reward + off_road_penalty + angle_penalty
        return total_reward

# 🎯 환경 감싸기 (Custom Wrapper 적용)
env = CustomCarRacingEnv(gym.make("CarRacing-v3", render_mode="human"))

# 저장된 모델이 있는지 확인 후 불러오기
if os.path.exists("sac_CarRacing.zip"):
    model = SAC.load("sac_CarRacing", env=env)  # 기존 모델 불러오기
    print("✅ 기존 학습된 모델 불러와서 추가 학습 진행")
else:
    model = SAC("MlpPolicy", env, verbose=1)  # 새로운 모델 생성
    print("🚀 저장된 모델이 없어 새로운 모델을 학습 시작")

# 추가 학습 진행 (이전 학습 내용을 유지하며 학습)
model.learn(total_timesteps=50000, log_interval=4)

# 모델 저장 (학습된 내용 유지)
model.save("sac_CarRacing")

# 테스트 실행
obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
