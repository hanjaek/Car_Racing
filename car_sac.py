import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# CarRacing 환경 생성
env = gym.make('CarRacing-v2', domain_randomize=False)  # 환경 색상 고정

# Monitor로 환경 감시 (로그 저장)
env = Monitor(env)

# 벡터 환경으로 래핑 (멀티프로세싱 지원)
env = DummyVecEnv([lambda: env])

# SAC 모델 초기화 (CNN 정책 사용)
model = SAC('CnnPolicy', env, verbose=1)

# 학습 수행 (최소 500만 스텝 권장)
model.learn(total_timesteps=5000000, log_interval=10)

# 모델 저장
model.save("sac_car_racing_best")

# 학습된 모델 로드 후 실행
del model
model = SAC.load("sac_car_racing_best")

obs = env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
