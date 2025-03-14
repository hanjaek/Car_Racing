import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# CarRacing 환경 생성 (환경 색상 고정)
env = gym.make("CarRacing-v2", domain_randomize=False, render_mode="rgb_array")

# Monitor로 환경 감시 (로그 저장)
env = Monitor(env)

# 벡터 환경으로 래핑 (SAC 학습 안정성을 위해 필요)
env = DummyVecEnv([lambda: env])

# SAC 모델 초기화 (CNN 정책 사용)
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
model.learn(total_timesteps=5000000, log_interval=10)

# 모델 저장
model.save("sac_car_racing_best")

# 학습된 모델 로드 후 실행
del model
model = SAC.load("sac_car_racing_best")

obs, _ = env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)  # 튜플 반환 값 처리
    env.render()
