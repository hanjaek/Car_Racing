import os
import gymnasium as gym
import numpy as np
import pygame
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.buffers import ReplayBuffer

# ✅ 커스텀 리플레이 버퍼 (사람 개입 우선 샘플링)
class HumanPrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.human_flags = np.zeros(self.buffer_size, dtype=bool)

    def _init_storage(self):
        super()._init_storage()
        self.human_flags = np.zeros(self.buffer_size, dtype=bool)

    def add(self, obs, next_obs, action, reward, done, infos):
        human_flag = infos[0].get("human", False)
        super().add(obs, next_obs, action, reward, done, infos)
        self.human_flags[self.pos - 1] = human_flag

    def sample(self, batch_size, env=None):
        size = int(self.size())
        human_indices = np.where(self.human_flags[:size])[0]
        other_indices = np.where(~self.human_flags[:size])[0]

        n_human = int(batch_size * 0.7)
        n_other = batch_size - n_human

        chosen = []
        if len(human_indices) > 0:
            chosen.extend(np.random.choice(human_indices, min(n_human, len(human_indices)), replace=False))
        if len(other_indices) > 0:
            chosen.extend(np.random.choice(other_indices, min(n_other, len(other_indices)), replace=False))

        chosen = np.array(chosen)
        return super()._get_samples(chosen, env=env)

# ✅ 설정
MODEL_DIR = "sac_hil_model"
LOG_DIR = "tensorboard_logs"
MODEL_PATH = os.path.join(MODEL_DIR, "sac_car_racing_best.zip")
BUFFER_PATH = os.path.join(MODEL_DIR, "replay_buffer.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

SEED = 1
MAX_HUMAN_STEPS = 100_000
TOTAL_STEPS = 1_000_000

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

# ✅ 모델 불러오기 or 새로 학습 시작
try:
    model = SAC.load(MODEL_PATH, env=env, tensorboard_log=LOG_DIR)
    model.load_replay_buffer(BUFFER_PATH)
    print("✅ 기존 모델 로드 완료.")
except:
    model = SAC(
        "CnnPolicy",
        env,
        replay_buffer_class=HumanPrioritizedReplayBuffer,
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
    print("🚀 새 모델로 학습 시작!")

# ✅ pygame 초기화
pygame.init()
screen = pygame.display.set_mode((300, 100))
pygame.display.set_caption("HIL Control")

# ✅ 조작 상태 변수
current_steering, current_speed = 0.0, 0.0
initial_alpha = 0.9
decay_rate = 0.5

def get_human_action(original_action, step):
    global current_steering, current_speed

    keys = pygame.key.get_pressed()
    action = np.array(original_action, dtype=np.float32).reshape(-1)
    steer_step = 0.1
    accel_step = 0.05
    brake_step = 0.1
    steering_recovery = 0.05

    if keys[pygame.K_LEFT]:
        current_steering -= steer_step
        action[2] = min(0.5, action[2] + brake_step)
    if keys[pygame.K_RIGHT]:
        current_steering += steer_step
        action[2] = min(0.5, action[2] + brake_step)
    if keys[pygame.K_UP]:
        current_speed += accel_step
        action[2] = 0.0
        current_steering -= np.sign(current_steering) * steering_recovery
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
obs = env.reset()
obs = obs.transpose(0, 3, 1, 2)
done = False
step = 0
human_intervened = False

while step < TOTAL_STEPS:
    pygame.event.pump()
    keys = pygame.key.get_pressed()

    action = model.predict(obs, deterministic=True)[0]
    human_override = False

    if step < MAX_HUMAN_STEPS and any(keys):
        action = get_human_action(action, step)
        human_override = True
        human_intervened = True

    action = action.reshape(1, -1)
    step_result = env.step(action)

    if len(step_result) == 4:
        next_obs, reward, done, info = step_result
        terminated, truncated = done, False
    else:
        next_obs, reward, terminated, truncated, info = step_result

    done = terminated or truncated
    next_obs = next_obs.transpose(0, 3, 1, 2)

    model.replay_buffer.add(
        obs, next_obs, action, [reward], [terminated], [{"human": human_override}]
    )

    if step < MAX_HUMAN_STEPS and step % 1000 == 0 and human_intervened:
        print(f"📢 Step {step}: Human-in-the-loop 학습 1000 스텝 진행...")
        model.learn(total_timesteps=1000, reset_num_timesteps=False)
        human_intervened = False

    obs = next_obs
    step += 1
    env.render()

    if step == MAX_HUMAN_STEPS:
        model.save(os.path.join(MODEL_DIR, "after_human_model.zip"))
        model.save_replay_buffer(BUFFER_PATH)

    if done:
        obs = env.reset()
        obs = obs.transpose(0, 3, 1, 2)
        current_steering, current_speed = 0.0, 0.0

print("🚀 사람 개입 데이터 기반 90만 스텝 학습 시작!")
model.learn(total_timesteps=TOTAL_STEPS - MAX_HUMAN_STEPS, reset_num_timesteps=False)
model.save(MODEL_PATH)
print("✅ 모델 저장 완료.")

pygame.quit()
