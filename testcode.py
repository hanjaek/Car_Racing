import os
import gymnasium as gym
import numpy as np
import pygame
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# 모델 저장할 폴더 경로 설정
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "sac_car_racing_best")

# 모델 폴더가 없으면 생성
os.makedirs(MODEL_DIR, exist_ok=True)

# CarRacing 환경 생성 (렌더링 가능하게 설정)
env = gym.make("CarRacing-v3", domain_randomize=False, render_mode="human")
env = Monitor(env)  # 환경 감시

# 학습된 모델 불러오기
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

# Pygame 초기화
pygame.init()
screen = pygame.display.set_mode((400, 300))  # 간단한 창 생성
pygame.display.set_caption("Human-in-the-Loop RL")

# 키보드 입력 변수
human_override = False
human_action = np.array([0.0, 0.0, 0.0])  # [steering, acceleration, brake]

def get_human_action():
    """ 키보드 입력을 받아 사람이 차량을 조작하는 액션을 반환 """
    global human_override, human_action
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    
    keys = pygame.key.get_pressed()
    
    if keys[pygame.K_LEFT]:
        human_action[0] = -1.0  # 좌회전
        human_override = True
    elif keys[pygame.K_RIGHT]:
        human_action[0] = 1.0  # 우회전
        human_override = True
    else:
        human_action[0] = 0.0  # 방향 유지

    if keys[pygame.K_UP]:
        human_action[1] = 1.0  # 가속
        human_override = True
    else:
        human_action[1] = 0.0  # 기본 상태
    
    if keys[pygame.K_DOWN]:
        human_action[2] = 0.8  # 브레이크
        human_override = True
    else:
        human_action[2] = 0.0
    
    return human_action if human_override else None

# 학습 수행 (최소 1000만 스텝)
TIMESTEPS = 10000000
obs = env.reset()

for step in range(TIMESTEPS):
    human_input = get_human_action()
    
    if human_input is not None:
        action = human_input  # 사람이 직접 조작
        print(f"🕹️ 인간이 조작 중! 액션: {action}")
    else:
        action, _states = model.predict(obs, deterministic=True)

    obs, reward, done, info = env.step(action)
    
    if done:
        obs = env.reset()
    
    # 사람이 개입한 경우, 해당 데이터를 학습 데이터로 저장
    if human_override:
        model.replay_buffer.add(obs, action, reward, done, obs)
        human_override = False  # 다시 RL이 학습하도록 초기화

env.close()
pygame.quit()
