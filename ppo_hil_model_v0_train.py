import os
import gymnasium as gym
import numpy as np
import pygame
import random
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# ------------------------ Seed 설정 ------------------------
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

# ------------------------ Pygame 초기화 ------------------------
pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("HIL Control Window")

# ------------------------ 디렉토리 설정 ------------------------
MODEL_DIR = "ppo_hil_model_v0_1"
LOG_DIR = "tensorboard_logs"
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_car_racing_best")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ------------------------ 환경 생성 함수 ------------------------
def make_env():
    def _init():
        env = gym.make("CarRacing-v3", domain_randomize=False, render_mode="human")
        env = Monitor(env, filename=os.path.join(LOG_DIR, "PPO_HIL_seed1_v0_1.csv"))
        env.reset(seed=SEED)
        return env
    return _init

# ------------------------ 환경 생성 ------------------------
env = DummyVecEnv([make_env()])
env.seed(SEED)

# ------------------------ 모델 로드 또는 새로 생성 ------------------------
try:
    model = PPO.load(MODEL_PATH, env=env, tensorboard_log=LOG_DIR)
    print(f"기존 모델 로드 완료: {MODEL_PATH}")
except:
    print("새 모델 생성 시작")
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

# ------------------------ 제어 변수 초기화 ------------------------
current_steering = 0.0
current_speed = 0.5

# ------------------------ HIL 하이퍼파라미터 ------------------------
initial_alpha = 0.9
min_alpha = 0.0
decay_rate = 0.1
max_human_steps = 10000  # 초기 개입만 수행

# ------------------------ 인간 개입 함수 ------------------------
def get_human_action(original_action, step):
    global current_steering, current_speed
    keys = pygame.key.get_pressed()
    action = np.array(original_action, dtype=np.float32).reshape(-1)

    steer_step = 0.1
    speed_step = 0.05
    brake_step = 0.1
    steering_recovery = 0.05

    if keys[pygame.K_LEFT]:
        current_steering -= steer_step
        action[2] = min(0.3, action[2] + brake_step)
    if keys[pygame.K_RIGHT]:
        current_steering += steer_step
        action[2] = min(0.3, action[2] + brake_step)
    if keys[pygame.K_UP]:
        current_speed += speed_step
        action[2] = 0.0
        if current_steering > 0:
            current_steering = max(0, current_steering - steering_recovery)
        elif current_steering < 0:
            current_steering = min(0, current_steering + steering_recovery)
    if keys[pygame.K_DOWN]:
        action[2] = 1.0
        current_speed *= 0.8
    if not any([keys[pygame.K_DOWN], keys[pygame.K_LEFT], keys[pygame.K_RIGHT]]):
        action[2] = max(0.0, action[2] - 0.05)

    current_steering = np.clip(current_steering, -1.0, 1.0)
    current_speed = np.clip(current_speed, 0.0, 1.0)
    action[2] = np.clip(action[2], 0.0, 1.0)

    alpha = max(min_alpha, initial_alpha - decay_rate * (step / max_human_steps)) if step < max_human_steps else 0.0
    action[0] = alpha * current_steering + (1 - alpha) * action[0]
    action[1] = alpha * current_speed + (1 - alpha) * action[1]
    action[1] = np.clip(action[1], 0.0, 1.0)

    return action

# ------------------------ 메인 루프 ------------------------
obs = env.reset()
obs = obs.transpose(0, 3, 1, 2)
step = 0
human_intervened = False

while step <= max_human_steps:
    pygame.event.pump()
    action = model.predict(obs, deterministic=True)[0]

    if step < max_human_steps and any(pygame.key.get_pressed()):
        action = get_human_action(action, step)
        human_intervened = True

    action = np.array(action).reshape(1, -1)
    result = env.step(action)

    if len(result) == 4:
        next_obs, reward, done, info = result
        terminated, truncated = done, False
    else:
        next_obs, reward, terminated, truncated, info = result

    done = terminated or truncated
    next_obs = next_obs.transpose(0, 3, 1, 2)

    if step % 2048 == 0 and human_intervened:
        model.learn(total_timesteps=2048, reset_num_timesteps=False)
        human_intervened = False

    obs = next_obs
    step += 1
    env.render()

    print(f"Step {step} | Human: {human_intervened} | Action: {action}")

    if done:
        current_steering, current_speed = 0.0, 0.0
        obs = env.reset()
        obs = obs.transpose(0, 3, 1, 2)

# ------------------------ 사람 개입 이후 전체 학습 ------------------------
print("사람 개입 데이터를 기반으로 전체 학습 수행")
model.save(os.path.join(MODEL_DIR, "after_human_model.zip"))
model.learn(total_timesteps=50000, reset_num_timesteps=False)
model.save(os.path.join(MODEL_DIR, "after_human_learned_model.zip"))
model.learn(total_timesteps=900000, reset_num_timesteps=False)
model.save(MODEL_PATH)

pygame.quit()
