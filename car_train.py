import os
import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# 모델 및 로그 저장할 폴더 경로 설정
MODEL_DIR = "model"
LOG_FILE = os.path.join(MODEL_DIR, "training_log.csv")
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
        buffer_size=100000,
        batch_size=64,
        tau=0.005,
        gamma=0.99,
        train_freq=4,
        gradient_steps=2,
        verbose=1
    )

# 학습 로그 저장을 위한 데이터 리스트
log_data = []

# 학습 수행 (최소 500만 스텝 권장)
for step in range(0, 5000000, 10000):  # 10,000 스텝마다 저장
    model.learn(total_timesteps=10000, reset_num_timesteps=False)
    
    # 학습 상태 확인
    training_info = {
        "total_timesteps": step + 10000,
        "ep_rew_mean": env.get_attr("episode_rewards")[-1] if env.get_attr("episode_rewards") else None,
        "actor_loss": model.actor.optimizer.param_groups[0]['lr'],
        "critic_loss": model.critic.optimizer.param_groups[0]['lr'],
        "ent_coef": model.ent_coef_optimizer.param_groups[0]['lr']
    }
    
    log_data.append(training_info)

    # 로그를 CSV 파일로 저장
    df = pd.DataFrame(log_data)
    df.to_csv(LOG_FILE, index=False)
    print(f"📊 로그 저장 완료: {LOG_FILE}")

# 모델 저장 (model 폴더 안에 저장됨)
model.save(MODEL_PATH)
print(f"💾 학습이 완료되었습니다. 모델이 저장되었습니다. ({MODEL_PATH})")
