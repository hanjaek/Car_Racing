import os
import csv
import gymnasium as gym
import numpy as np
import pygame  
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# ✅ Pygame 초기화
pygame.init()
screen = pygame.display.set_mode((400, 300))  
pygame.display.set_caption("HIL Control Window")

# ✅ 모델 및 로그 저장 폴더 설정
MODEL_DIR = "test"
LOG_DIR = "tensorboard_logs"
MODEL_PATH = os.path.join(MODEL_DIR, "sac_car_racing_best")
CSV_FILE = "test.csv"  # ✅ CSV 파일 이름

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ✅ CarRacing 환경 생성
env = gym.make("CarRacing-v3", domain_randomize=False, render_mode="human")
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

# ✅ 초기 속도 및 방향 변수
current_steering = 0.0
current_speed = 0.0

def get_human_action(original_action):
    global current_steering, current_speed
    keys = pygame.key.get_pressed()
    action = np.array(original_action, dtype=np.float32).reshape(-1)  

    steer_step = 0.1
    speed_step = 0.05
    steering_recovery = 0.05

    if keys[pygame.K_LEFT]:  
        current_steering -= steer_step  
    if keys[pygame.K_RIGHT]:  
        current_steering += steer_step  
    if keys[pygame.K_UP]:  
        current_speed += speed_step
        if current_steering > 0:
            current_steering = max(0, current_steering - steering_recovery)
        elif current_steering < 0:
            current_steering = min(0, current_steering + steering_recovery)
    if keys[pygame.K_DOWN]:  
        current_speed -= speed_step  

    current_steering = np.clip(current_steering, -1.0, 1.0)
    current_speed = np.clip(current_speed, 0.0, 1.0)  

    action[0] = current_steering  
    action[1] = current_speed  
    action[2] = 0.0  

    return action

# ✅ HIL 학습 루프 (300만 스텝)
obs = env.reset()
step = 0
total_timesteps = 3000000

# ✅ CSV 파일 생성 및 헤더 작성
with open(CSV_FILE, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Step", "Human Override", "Steering", "Acceleration", "Brake"])  # ✅ 헤더 추가

try:
    while step < total_timesteps:
        pygame.event.pump()
        human_override = False  
        action = model.predict(obs, deterministic=True)[0]

        if any(pygame.key.get_pressed()):  
            action = get_human_action(action)
            human_override = True  

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # ✅ CSV에 데이터 저장
        with open(CSV_FILE, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                step, 
                human_override, 
                action[0][0],  # Steering
                action[0][1],  # Acceleration
                action[0][2]   # Brake
            ])

        # ✅ 자동 주행 데이터 학습 계속 진행
        model.replay_buffer.add(
            np.array(obs),  
            np.array(next_obs),  
            np.array(action),  
            np.array([reward]),  
            np.array([terminated]),  
            [{}]  
        )

        # ✅ 1000 스텝마다 SAC 학습 실행 (사람 개입 여부와 관계없이)
        if step % 1000 == 0:
            model.learn(total_timesteps=1000)

        obs = next_obs
        step += 1

        print(f"Step: {step}, Human Override: {human_override}, Action: {action}")

except Exception as e:
    print(f"🚨 오류 발생: {e}")
finally:
    print(f"💾 학습 로그가 '{CSV_FILE}'에 저장되었습니다.")
    pygame.quit()
