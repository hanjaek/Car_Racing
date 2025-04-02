import os
import gymnasium as gym
import numpy as np
import random
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# 모델 및 로그 저장할 폴더 설정
MODEL_DIR = "ppo_basic_model_1"
LOG_DIR = "tensorboard_logs"
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_car_racing_best")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# SEED 값 설정
SEED = 1

# 시드 고정
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 환경 생성 함수
def make_env():
    def _init():
        env = gym.make("CarRacing-v3", domain_randomize=False, render_mode="human")
        env = Monitor(env, filename=os.path.join(LOG_DIR, "ppo_seed1_1.csv"))
        env.reset(seed=SEED)
        return env
    return _init

# CarRacing 환경 생성 (SEED 적용)
env = DummyVecEnv([make_env()])
env.seed(SEED)

# 기존 모델 불러오기 or 새로운 모델 생성
try:
    model = PPO.load(MODEL_PATH, env=env, tensorboard_log=LOG_DIR)
    print(f"✅ 기존 모델을 불러와서 추가 학습합니다. ({MODEL_PATH})")
except:
    print("🚀 기존 모델이 없어서 새로 학습을 시작합니다.")
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=LOG_DIR,
        seed=SEED  
    )

# 학습 수행 (항상 동일한 트랙에서 학습)
model.learn(total_timesteps=1000000, log_interval=10)

# 모델 저장
model.save(MODEL_PATH)
print(f"💾 학습이 완료되었습니다. 모델이 '{MODEL_PATH}'에 저장되었습니다.")
