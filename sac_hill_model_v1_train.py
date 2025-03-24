import os
import gymnasium as gym
import numpy as np
import pygame
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.buffers import ReplayBuffer

# ✅ 사람 개입 데이터를 별도로 저장하는 버퍼
class SeparateHumanReplayBuffer:
    def __init__(self, buffer_size, obs_shape, action_dim):
        self.size = buffer_size
        self.ptr = 0
        self.full = False

        self.obs = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=bool)

    def add(self, obs, next_obs, action, reward, done):
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.size
        self.full = self.full or self.ptr == 0

    def sample(self, batch_size):
        max_idx = self.size if self.full else self.ptr
        indices = np.random.choice(max_idx, batch_size, replace=False)
        return {
            "observations": self.obs[indices],
            "actions": self.actions[indices],
            "next_observations": self.next_obs[indices],
            "rewards": self.rewards[indices],
            "dones": self.dones[indices]
        }

# ✅ 설정
MODEL_DIR = "sac_hil_model_final"
LOG_DIR = "tensorboard_logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

SEED = 1
MAX_HUMAN_STEPS = 100_000
TOTAL_STEPS = 1_000_000
HUMAN_BUFFER_SIZE = 100_000
HUMAN_BATCH_RATIO = 0.5  # 50%는 무조건 human

# ✅ 환경 생성
def make_env():
    def _init():
        env = gym.make("CarRacing-v3", domain_randomize=False, render_mode="human")
        env = Monitor(env)
        env.reset(seed=SEED)
        return env
    return _init

env = DummyVecEnv([make_env()])
env.seed(SEED)

model = SAC(
    "CnnPolicy",
    env,
    buffer_size=1_000_000,
    learning_rate=3e-4,
    batch_size=256,
    train_freq=1,
    gradient_steps=1,
    tau=0.005,
    gamma=0.99,
    verbose=1,
    tensorboard_log=LOG_DIR
)

# ✅ 사람 버퍼 준비
obs_shape = (3, 96, 96)
human_buffer = SeparateHumanReplayBuffer(HUMAN_BUFFER_SIZE, obs_shape, 3)

# ✅ pygame 초기화
pygame.init()
screen = pygame.display.set_mode((300, 100))
pygame.display.set_caption("HIL Control")

current_steering, current_speed = 0.0, 0.0
initial_alpha = 0.9
decay_rate = 0.5

def get_human_action(original_action, step):
    global current_steering, current_speed
    keys = pygame.key.get_pressed()
    action = np.array(original_action, dtype=np.float32).reshape(-1)
    if keys[pygame.K_LEFT]:
        current_steering -= 0.1
        action[2] = min(0.5, action[2] + 0.1)
    if keys[pygame.K_RIGHT]:
        current_steering += 0.1
        action[2] = min(0.5, action[2] + 0.1)
    if keys[pygame.K_UP]:
        current_speed += 0.05
        action[2] = 0.0
        current_steering -= np.sign(current_steering) * 0.05
    if keys[pygame.K_DOWN]:
        action[2] = 1.0
        current_speed *= 0.8
    if not any([keys[pygame.K_LEFT], keys[pygame.K_RIGHT], keys[pygame.K_DOWN]]):
        action[2] = max(0.0, action[2] - 0.05)
    current_steering = np.clip(current_steering, -1, 1)
    current_speed = np.clip(current_speed, 0, 1)
    action[2] = np.clip(action[2], 0, 1)
    alpha = max(0.0, initial_alpha - decay_rate * (step / MAX_HUMAN_STEPS)) if step < MAX_HUMAN_STEPS else 0.0
    action[0] = alpha * current_steering + (1 - alpha) * action[0]
    action[1] = alpha * current_speed + (1 - alpha) * action[1]
    return action

# ✅ 학습 루프
obs = env.reset().transpose(0, 3, 1, 2)
done = False
step = 0

while step < TOTAL_STEPS:
    pygame.event.pump()
    keys = pygame.key.get_pressed()
    action = model.predict(obs, deterministic=True)[0]
    human_override = False

    if step < MAX_HUMAN_STEPS and any(keys):
        action = get_human_action(action, step)
        human_override = True

    reshaped_action = action.reshape(1, -1)
    step_result = env.step(reshaped_action)
    if len(step_result) == 5:
        next_obs, reward, terminated, truncated, info = step_result
    else:
        next_obs, reward, done, info = step_result
        terminated, truncated = done, False

    done = terminated or truncated
    next_obs = next_obs.transpose(0, 3, 1, 2)

    # 저장
    if human_override:
        human_buffer.add(obs[0], next_obs[0], action, reward, terminated)
    else:
        model.replay_buffer.add(obs, next_obs, reshaped_action, [reward], [terminated], [{}])

    # 학습
    if step > 1000:
        human_batch = human_buffer.sample(int(model.batch_size * HUMAN_BATCH_RATIO))
        agent_batch = model.replay_buffer.sample(model.batch_size - int(model.batch_size * HUMAN_BATCH_RATIO))

        total_batch = {
            "observations": np.concatenate([human_batch["observations"], agent_batch.observations]),
            "actions": np.concatenate([human_batch["actions"], agent_batch.actions]),
            "next_observations": np.concatenate([human_batch["next_observations"], agent_batch.next_observations]),
            "rewards": np.concatenate([human_batch["rewards"], agent_batch.rewards]),
            "dones": np.concatenate([human_batch["dones"], agent_batch.dones]),
        }
        model.train(batch=total_batch)

    obs = next_obs
    step += 1
    env.render()

    if done:
        obs = env.reset().transpose(0, 3, 1, 2)
        current_steering, current_speed = 0.0, 0.0

    if step % 10000 == 0:
        model.save(os.path.join(MODEL_DIR, f"checkpoint_step_{step}.zip"))

print("✅ 전체 학습 완료")
model.save(os.path.join(MODEL_DIR, "sac_car_racing_best.zip"))
pygame.quit()