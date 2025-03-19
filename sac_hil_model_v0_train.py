import os
import gymnasium as gym
import pygame
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# ✅ 모델 및 로그 저장할 폴더 설정
MODEL_DIR = "basic_model_v0"
LOG_DIR = "tensorboard_logs"  
MODEL_PATH = os.path.join(MODEL_DIR, "sac_car_racing_best")

# 폴더가 없으면 생성
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ✅ CarRacing 환경 생성
env = gym.make("CarRacing-v3", domain_randomize=False, render_mode="human")
env = Monitor(env)
env = DummyVecEnv([lambda: env])

# ✅ 키보드 입력 설정
pygame.init()
screen = pygame.display.set_mode((400, 300))  # PyGame 창 (렌더링 없이 입력만 받음)
pygame.display.set_caption("Human-in-the-Loop Controller")

# 키보드 입력을 행동 값으로 변환하는 함수
def get_human_action():
    keys = pygame.key.get_pressed()
    
    action = np.array([0.0, 0.0, 0.0])  # [steering, acceleration, brake]

    if keys[pygame.K_LEFT]:   # 좌회전
        action[0] = -1.0  
    if keys[pygame.K_RIGHT]:  # 우회전
        action[0] = 1.0  
    if keys[pygame.K_UP]:     # 가속
        action[1] = 1.0  
    if keys[pygame.K_DOWN]:   # 브레이크
        action[2] = 1.0  

    return action

# ✅ 기존 모델 불러오기 or 새로운 모델 생성
try:
    model = SAC.load(MODEL_PATH, env=env, tensorboard_log=LOG_DIR)
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
        verbose=1,
        tensorboard_log=LOG_DIR
    )

# ✅ Human-in-the-Loop 학습 루프
TIMESTEPS = 3000000
obs = env.reset()
human_override = False

for step in range(TIMESTEPS):
    pygame.event.pump()  # 이벤트 처리
    human_action = get_human_action()

    # 사람이 입력했는지 체크
    if np.any(human_action != 0.0):
        action = human_action  # 사람이 조작한 행동을 사용
        human_override = True
    else:
        action, _states = model.predict(obs, deterministic=True)  # AI가 행동 결정
        human_override = False

    obs, reward, done, info = env.step(action)

    # 사람이 조작한 데이터는 별도 버퍼에 저장하여 학습 데이터로 활용 가능
    if human_override:
        model.replay_buffer.add(obs, action, reward, done, obs)

    if done:
        obs = env.reset()

# ✅ 모델 저장
model.save(MODEL_PATH)
print(f"💾 학습이 완료되었습니다. 모델이 '{MODEL_PATH}'에 저장되었습니다.")
pygame.quit()
