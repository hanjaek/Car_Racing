import gymnasium as gym
from stable_baselines3 import SAC

env = gym.make("CarRacing-v3", render_mode="human")

# 저장된 모델이 있는지 확인 후 불러오기
try:
    model = SAC.load("sac_CarRacing", env=env)
    print("✅ 기존 학습된 모델 불러옴. 추가 학습 진행.")
except:
    print("🚀 저장된 모델이 없어 새로운 모델을 학습합니다.")
    model = SAC("MlpPolicy", env, verbose=1)

# 추가 학습 진행
model.learn(total_timesteps=10000, log_interval=4)

# 모델 저장
model.save("sac_CarRacing")

# 테스트 실행
obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
