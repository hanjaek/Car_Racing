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
env = gym.make("CarRacing-v3", domain_randomize=False, render_mode="human")  # 🚀 화면 출력 가능하게 변경
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

# ✅ 키 입력을 받아 사람이 개입할 수 있도록 하는 함수
def get_human_action(original_action):
    keys = pygame.key.get_pressed()
    
    # ✅ 모델이 예측한 기존 행동을 기반으로 조정 (NumPy 배열로 변환)
    action = np.array(original_action, dtype=np.float32)
    step = 0.1  # 키 입력에 따른 조절 강도 (부드러운 조작)
    
    if keys[pygame.K_LEFT]:   # ← 왼쪽 방향키
        action[0] -= step
    if keys[pygame.K_RIGHT]:  # → 오른쪽 방향키
        action[0] += step
    if keys[pygame.K_UP]:     # ↑ 가속
        action[1] += step
    if keys[pygame.K_DOWN]:   # ↓ 브레이크
        action[2] += step

    # ✅ NumPy 배열 형태 유지하면서 값 제한
    action = np.clip(action, [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0])

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
        action = get_human_action(action)
        human_override = True  # 사람이 개입했음을 표시

    # ✅ 환경 업데이트 (Gymnasium step() 반환값 처리)
    step_result = env.step(action)

    if len(step_result) == 4:  # (next_obs, reward, done, info) 반환하는 경우
        next_obs, reward, done, info = step_result
        terminated, truncated = done, False  # `done`을 terminated로 사용하고, truncated는 False로 설정
    elif len(step_result) == 5:  # (next_obs, reward, terminated, truncated, info) 반환하는 경우
        next_obs, reward, terminated, truncated, info = step_result
    else:
        raise ValueError(f"Unexpected number of return values from env.step(action): {len(step_result)}")

    done = terminated or truncated

    # 사람이 개입한 경우만 모델 학습 데이터로 추가
    if human_override:
        model.replay_buffer.add(obs, next_obs, action, reward, terminated, [{}])

    obs = next_obs  # 다음 상태로 업데이트
    step += 1
    env.render()

# ✅ 모델 저장
model.save(MODEL_PATH)
print(f"💾 학습이 완료되었습니다. 모델이 '{MODEL_PATH}'에 저장되었습니다.")

pygame.quit()
