import os
import gymnasium as gym
import numpy as np
import pygame  # ✅ 키보드 입력을 처리하기 위한 라이브러리
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# ✅ Pygame 초기화 (HIL을 위한 키 입력 처리)
pygame.init()
screen = pygame.display.set_mode((400, 300))  # Pygame 창 (실제 게임 화면과는 무관)
pygame.display.set_caption("HIL Control Window")

# ✅ 모델 및 로그 저장할 폴더 설정
MODEL_DIR = "sac_hil_model_v0"
LOG_DIR = "tensorboard_logs"  
MODEL_PATH = os.path.join(MODEL_DIR, "sac_car_racing_best")

# 폴더가 없으면 생성
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ✅ CarRacing 환경 생성
env = gym.make("CarRacing-v3", domain_randomize=False, render_mode="rgb_array")
env = Monitor(env)
env = DummyVecEnv([lambda: env])

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

# ✅ 키 입력을 받아서 사람이 개입할 수 있도록 하는 함수
def get_human_action():
    keys = pygame.key.get_pressed()
    
    # 기본적으로 모델의 출력을 사용
    action = np.array([0.0, 0.0, 0.0])  # [steering, gas, brake]

    if keys[pygame.K_LEFT]:   # ← 왼쪽 방향키
        action[0] = -1.0
    if keys[pygame.K_RIGHT]:  # → 오른쪽 방향키
        action[0] = 1.0
    if keys[pygame.K_UP]:     # ↑ 가속
        action[1] = 1.0
    if keys[pygame.K_DOWN]:   # ↓ 브레이크
        action[2] = 0.8  # 브레이크는 1.0보다 약하게 조정
    
    return action

# ✅ HIL 학습 루프 (300만 스텝)
obs = env.reset()  
done = False
total_timesteps = 3000000
step = 0

while step < total_timesteps:
    pygame.event.pump()  # 키보드 입력을 갱신
    
    human_override = False  # 사람이 개입했는지 여부
    action = model.predict(obs, deterministic=True)[0]  # 기본적으로 모델 행동 사용

    if any(pygame.key.get_pressed()):  # 사람이 키를 누르면 HIL 개입
        action = get_human_action()
        human_override = True  # 사람이 개입했음을 표시

    # 환경 업데이트
    next_obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    
    # 사람이 개입한 경우, 모델이 학습할 수 있도록 버퍼에 추가
    if human_override:
        model.replay_buffer.add(obs, action, reward, next_obs, terminated)

    obs = next_obs  # 다음 상태로 업데이트
    step += 1
    env.render()

# ✅ 모델 저장
model.save(MODEL_PATH)
print(f"💾 학습이 완료되었습니다. 모델이 '{MODEL_PATH}'에 저장되었습니다.")

pygame.quit()
