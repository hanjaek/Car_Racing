import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

MODEL_DIR = "sac_hil_model"
MODEL_PATH = os.path.join(MODEL_DIR, "sac_hil_model")

SEED = 1  

def make_env():
    def _init():
        env = gym.make("CarRacing-v3", domain_randomize=False, render_mode="human")
        env.reset(seed=SEED) 
        return env
    return _init

env = DummyVecEnv([make_env()])
env.seed(SEED)

model = SAC.load(MODEL_PATH)
print(f"학습된 모델을 불러왔습니다. ({MODEL_PATH}) 자율주행을 시작합니다.")

obs = env.reset()
env.render()  

done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)  
    obs, reward, done, info = env.step(action)  

    env.render()
