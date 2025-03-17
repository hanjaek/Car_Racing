import os
import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter

# 📌 모델 & 로그 저장 경로
MODEL_DIR = "basic_model_v0"
MODEL_PATH = os.path.join(MODEL_DIR, "sac_car_racing_best")
LOG_DIR = "tensorboard_logs/basic_model_v0_logs"

# 폴더 생성
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# TensorBoard 기록기 생성
writer = SummaryWriter(LOG_DIR)

# CarRacing 환경 생성
env = gym.make("CarRacing-v3", domain_randomize=False, render_mode="rgb_array")
env = Monitor(env)
env = DummyVecEnv([lambda: env])

# 모델 불러오기
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

# 🏆 학습 모니터링을 위한 지표 기록 함수
def log_training_metrics(model, writer, num_episodes=10):
    episode_rewards = []
    episode_lengths = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = [False]
        total_reward = 0
        step = 0

        while not done[0]:  
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            step += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(step)

        # 🟢 TensorBoard에 다양한 지표 기록
        writer.add_scalar("Rewards/Total Reward per Episode", total_reward, ep)
        writer.add_scalar("Episode_Length/Steps per Episode", step, ep)

        if ep % 10 == 0:
            writer.flush()

        print(f"Episode {ep+1}: Total Reward = {total_reward}, Steps = {step}")

    # 추가로 평균 보상도 기록
    mean_reward = np.mean(episode_rewards)
    writer.add_scalar("Rewards/Mean Reward", mean_reward)

    return episode_rewards

# 🔥 학습 진행 (학습 중에도 보상 저장)
for i in range(10):  
    model.learn(total_timesteps=100000, reset_num_timesteps=False)
    model.save(MODEL_PATH)
    print(f"💾 {((i+1)*100000)} 스텝 진행됨. 모델이 저장되었습니다. ({MODEL_PATH})")

    # 학습 중에도 지표 기록
    log_training_metrics(model, writer, num_episodes=10)

# TensorBoard 기록 종료
writer.close()
print(f"📊 TensorBoard 로그가 '{LOG_DIR}'에 저장되었습니다. 실행하려면:")
print(f"tensorboard --logdir={LOG_DIR}")
