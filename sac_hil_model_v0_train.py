import os
import gymnasium as gym
import numpy as np
import pygame
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Pygame 초기화
pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("HIL Control Window")

# 디렉토리 설정
MODEL_DIR = "sac_hil_model_v0"
LOG_DIR = "tensorboard_logs"
MODEL_PATH = os.path.join(MODEL_DIR, "sac_car_racing_best")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Seed 설정
SEED = 1

def make_env():
    def _init():
        env = gym.make("CarRacing-v3", domain_randomize=False, render_mode="human")
        env = Monitor(env, filename=os.path.join(LOG_DIR, "SAC_HIL_seed1.csv"))
        env.reset(seed=SEED)
        return env
    return _init

# 환경 생성
env = DummyVecEnv([make_env()])
env.seed(SEED)

# 기존 모델 불러오기 또는 새 모델 생성
try:
    model = SAC.load(MODEL_PATH, env=env, tensorboard_log=LOG_DIR)
    print(f"✅ 기존 모델 로드 완료: {MODEL_PATH}")
except:
    print("🚀 새 모델 생성 시작")
    model = SAC(
        "CnnPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=LOG_DIR
    )

# 사람 조작 관련 변수
current_steering = 0.0
current_speed = 0.0

# HIL 하이퍼파라미터
initial_alpha = 0.9
min_alpha = 0.0
decay_rate = 0.5
max_human_steps = 100000

# 키보드 입력 기반 사람 개입 액션 생성
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

    return action

# 1000 스텝마다 사람 개입 있으면 학습
def train_if_human_intervened(step):
    global human_intervened
    if step < max_human_steps and step % 1000 == 0 and human_intervened:
        print(f"📢 Step {step}: 사람 개입 → 1000 스텝 학습")
        model.learn(total_timesteps=1000, reset_num_timesteps=False)
        human_intervened = False

# 메인 루프
obs = env.reset()
obs = obs.transpose(0, 3, 1, 2)
step = 0
total_timesteps = 1_000_000
human_intervened = False

while step < total_timesteps:
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

    # 리플레이 버퍼에 transition 추가
    model.replay_buffer.add(obs, next_obs, action, [reward], [terminated], [{}])

    train_if_human_intervened(step)

    obs = next_obs
    step += 1
    env.render()

    print(f"Step {step} | Human: {human_intervened} | Action: {action}")

    if step == max_human_steps:
        print("💾 모델 저장 (사람 개입 종료 시점)")
        model.save(os.path.join(MODEL_DIR, "after_human_model.zip"))

    if done:
        current_steering, current_speed = 0.0, 0.0
        obs = env.reset()
        obs = obs.transpose(0, 3, 1, 2)

# 사람 개입 기반으로 후속 학습 (90만 스텝)
print("🚀 사람 개입 데이터를 기반으로 90만 스텝 학습 시작")
model = SAC.load(os.path.join(MODEL_DIR, "after_human_model.zip"), env=env, tensorboard_log=LOG_DIR)
model.learn(total_timesteps=900_000, reset_num_timesteps=False)

model.save(MODEL_PATH)
print(f"✅ 학습 완료! 최종 모델 저장됨 → {MODEL_PATH}")

pygame.quit()