import os
import gymnasium as gym
import numpy as np
import pygame
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# ------------------------ Pygame 초기화 ------------------------
pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("HIL Control Window")

# ------------------------ 디렉토리 설정 ------------------------
MODEL_DIR = "sac_hil_model_final"
LOG_DIR = "tensorboard_logs"
MODEL_PATH = os.path.join(MODEL_DIR, "sac_car_racing_best")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ------------------------ Seed 설정 ------------------------
SEED = 1

# ------------------------ 환경 생성 함수 ------------------------
def make_env():
    def _init():
        env = gym.make("CarRacing-v3", domain_randomize=False, render_mode="human")
        env = Monitor(env, filename=os.path.join(LOG_DIR, "SAC_HIL.csv"))
        env.reset(seed=SEED)
        return env
    return _init

# ------------------------ 환경 생성 ------------------------
env = DummyVecEnv([make_env()])
env.seed(SEED)

# ------------------------ 모델 로드 또는 새로 생성 ------------------------
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

# ------------------------ 변수 초기화 ------------------------
current_steering = 0.0
current_speed = 0.0
initial_alpha = 0.9
min_alpha = 0.0
decay_rate = 0.1
max_human_steps = 50000

# ------------------------ 사람 개입 함수 ------------------------
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
    current_speed = np.clip(current_speed, 0.0, 0.7)
    action[2] = np.clip(action[2], 0.0, 1.0)

    alpha = max(min_alpha, initial_alpha - decay_rate * (step / max_human_steps)) if step < max_human_steps else 0.0
    action[0] = alpha * current_steering + (1 - alpha) * action[0]
    action[1] = alpha * current_speed + (1 - alpha) * action[1]
    action[1] = np.clip(action[1], 0.0, 0.7)

    return action

# ------------------------ 메인 루프 ------------------------
obs = env.reset()
obs = obs.transpose(0, 3, 1, 2)
step = 0
human_intervened = False
reward_buffer = []
length_buffer = []
current_ep_reward = 0
current_ep_length = 0

while step <= max_human_steps:
    pygame.event.pump()
    action = model.predict(obs, deterministic=True)[0]

    if any(pygame.key.get_pressed()):
        action = get_human_action(action, step)
        human_intervened = True

    action = np.array(action).reshape(1, -1)
    result = env.step(action)

    if len(result) == 4:
        next_obs, reward, done, info = result
        terminated, truncated = done, False
    else:
        next_obs, reward, terminated, truncated, info = result

    if action[0][2] > 0.5 and current_speed < 0.2:
        reward -= 0.1
    elif action[0][2] > 0.6:
        reward -= 0.01

    done = terminated or truncated
    next_obs = next_obs.transpose(0, 3, 1, 2)

    # ReplayBuffer에 transition 저장
    model.replay_buffer.add(obs, next_obs, action, [reward], [terminated], [{}])

    # 콘솔 출력 (오류 방지용 .item() 사용)
    print(
        f"[STEP {step}] "
        f"Reward: {float(reward):.2f} | "
        f"Steering: {float(action[0][0]):.2f} | "
        f"Speed: {float(action[0][1]):.2f} | "
        f"Brake: {float(action[0][2]):.2f}"
    )

    obs = next_obs
    step += 1
    current_ep_reward += reward
    current_ep_length += 1
    env.render()

    if step % 1000 == 0 and human_intervened:
        print(f"📢 Step {step}: 사람 개입 → SAC 학습 (1000 스텝)")
        model.learn(total_timesteps=1000, reset_num_timesteps=False)
        human_intervened = False

    if done:
        reward_buffer.append(current_ep_reward)
        length_buffer.append(current_ep_length)
        model.logger.record("rollout/ep_rew_mean", np.mean(reward_buffer[-10:]))
        model.logger.record("rollout/ep_len_mean", np.mean(length_buffer[-10:]))
        model.logger.dump(step=step)

        current_ep_reward = 0
        current_ep_length = 0
        current_steering, current_speed = 0.0, 0.0
        obs = env.reset()
        obs = obs.transpose(0, 3, 1, 2)

# ------------------------ 모델 저장 ------------------------
print("✅ 사람 개입 학습 종료, 모델 저장 중...")
model.save(MODEL_PATH)
print(f"💾 모델 저장 완료: {MODEL_PATH}")

pygame.quit()
