import os
import gymnasium as gym
import numpy as np
import pygame
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.buffers import ReplayBuffer

# ✅ 커스텀 버퍼: 사람 개입 데이터를 일정 비율로 섞어주는 샘플링
class HILReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, human_ratio=0.5, **kwargs):
        super().__init__(buffer_size, observation_space, action_space, **kwargs)
        self.human_ratio = human_ratio

    def add(self, obs, next_obs, action, reward, done, infos):
        is_human = infos[0].get("human", False)
        super().add(obs, next_obs, action, reward, done, infos)
        self.human_flags[self.pos - 1] = is_human

    def sample(self, batch_size, env=None):
        size = int(self.size())
        human_idx = np.where(self.human_flags[:size])[0]
        other_idx = np.where(~self.human_flags[:size])[0]

        n_human = int(batch_size * self.human_ratio)
        n_agent = batch_size - n_human

        idx = []
        if len(human_idx) > 0:
            idx += list(np.random.choice(human_idx, min(n_human, len(human_idx)), replace=False))
        if len(other_idx) > 0:
            idx += list(np.random.choice(other_idx, min(n_agent, len(other_idx)), replace=False))

        idx = np.array(idx)
        return super()._get_samples(idx, env=env)

# ✅ 설정
MODEL_DIR = "sac_hil_model_v1"
os.makedirs(MODEL_DIR, exist_ok=True)

SEED = 1
MAX_HUMAN_STEPS = 100_000
TOTAL_STEPS = 1_000_000

# ✅ 환경 생성 함수
def make_env():
    def _init():
        env = gym.make("CarRacing-v3", domain_randomize=False, render_mode="human")
        env = Monitor(env)
        env.reset(seed=SEED)
        return env
    return _init

env = DummyVecEnv([make_env()])
env.seed(SEED)

# ✅ 모델 생성
model = SAC(
    "CnnPolicy",
    env,
    replay_buffer_class=HILReplayBuffer,
    replay_buffer_kwargs={"human_ratio": 0.5},  #
    buffer_size=1_000_000,                   
    learning_rate=3e-4,
    batch_size=256,
    train_freq=1,
    gradient_steps=1,
    tau=0.005,
    gamma=0.99,
    verbose=1,
    tensorboard_log="tensorboard_logs"
)


# ✅ Pygame 초기화
pygame.init()
screen = pygame.display.set_mode((400, 100))
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
obs = env.reset().transpose(0, 3, 1, 2)
step = 0
while step < TOTAL_STEPS:
    pygame.event.pump()
    keys = pygame.key.get_pressed()

    action = model.predict(obs, deterministic=True)[0]
    human_override = False

    if step < MAX_HUMAN_STEPS and any(keys):
        action = get_human_action(action, step)
        human_override = True

    action = action.reshape(1, -1)
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    next_obs = next_obs.transpose(0, 3, 1, 2)

    # ✅ 버퍼에 저장 (human 여부 표시)
    model.replay_buffer.add(
        obs, next_obs, action, [[float(reward)]], [[done]], [{"human": human_override}]
    )

    # ✅ 학습
    if step > 1000:
        model.train(batch_size=model.batch_size, gradient_steps=1)

    obs = next_obs
    step += 1
    env.render()

    if done:
        obs = env.reset().transpose(0, 3, 1, 2)
        current_steering, current_speed = 0.0, 0.0

    if step % 1000 == 0:
        print(f"[{step:06d}] Human: {human_override} | Reward: {float(reward):.2f} | Action: {action}")

    if step % 10000 == 0:
        model.save(os.path.join(MODEL_DIR, f"checkpoint_{step}.zip"))

# ✅ 종료 후 저장
model.save(os.path.join(MODEL_DIR, "sac_hil_model_v1.zip"))
pygame.quit()
