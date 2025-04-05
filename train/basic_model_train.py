import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

MODEL_DIR = "../basic_model"
LOG_DIR = "../tensorboard_logs"  
MODEL_PATH = os.path.join(MODEL_DIR, "basic_model")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

SEED = 1  

def make_env():
    def _init():
        env = gym.make("CarRacing-v3", domain_randomize=False, render_mode="rgb_array")
        env = Monitor(env, filename=os.path.join(LOG_DIR, "basic_model.csv"))  
        env.reset(seed=SEED) 
        return env
    return _init

env = DummyVecEnv([make_env()])
env.seed(SEED)  

try:
    model = SAC.load(MODEL_PATH, env=env, tensorboard_log=LOG_DIR)
    print(f"- 기존 모델을 불러와서 추가 학습합니다. ({MODEL_PATH})")
except:
    print("- 기존 모델이 없어서 새로 학습을 시작합니다.")
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
        verbose=1,
        tensorboard_log=LOG_DIR  
    )

model.learn(total_timesteps=1000000, log_interval=10)

model.save(MODEL_PATH)
print(f"- 학습이 완료되었습니다. 모델이 '{MODEL_PATH}'에 저장되었습니다.")
