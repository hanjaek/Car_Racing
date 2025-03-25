import os
import gymnasium as gym
import numpy as np
import pygame
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.buffers import ReplayBuffer
import pickle


# ============================== 설정 ==============================

SEED = 1
MODEL_DIR = "sac_hil_model_v1"
LOG_DIR = "tensorboard_logs"
MODEL_PATH = os.path.join(MODEL_DIR, "sac_car_racing_best")
HUMAN_MODEL_PATH = os.path.join(MODEL_DIR, "after_human_model.zip")
HUMAN_BUFFER_PATH = os.path.join(MODEL_DIR, "human_buffer.pkl")
AGENT_BUFFER_PATH = os.path.join(MODEL_DIR, "agent_buffer.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ============================== 환경 설정 ==============================
class CnnReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device,
        n_envs=1,
        optimize_memory_usage=False,
        handle_timeout_termination=True,
    ):
        # 수동으로 shape 지정
        self.image_obs = True
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )

    def _initialize_buffers(self):
        # 이미지 관측값에 맞게 obs 버퍼 초기화
        self.observations = np.zeros(
            (self.buffer_size, *self.observation_space.shape), dtype=np.uint8
        )
        self.next_observations = np.zeros(
            (self.buffer_size, *self.observation_space.shape), dtype=np.uint8
        )
        self.actions = np.zeros(
            (self.buffer_size, self.action_dim), dtype=np.float32
        )
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size,), dtype=np.float32)
        self.timeouts = np.zeros((self.buffer_size,), dtype=np.float32)  # for timeout terminations
        self.pos = 0
        self.full = False

    def _normalize_obs(self, obs):
        return obs  # 정규화는 하지 않음

    def sample(self, batch_size, env=None):
        return super().sample(batch_size, env)

def make_env():
    def _init():
        env = gym.make("CarRacing-v3", domain_randomize=False, render_mode="human")
        env = Monitor(env, filename=os.path.join(LOG_DIR, "SAC_HIL_ceed1.csv"))
        env.reset(seed=SEED)
        return env
    return _init

env = DummyVecEnv([make_env()])
env.seed(SEED)

def save_buffer(buffer, path):
    with open(path, 'wb') as f:
        pickle.dump(buffer, f)

def load_buffer(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# ============================== 모델 로드 ==============================

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
        tensorboard_log=LOG_DIR,
        seed=SEED
    )


model.learn(total_timesteps=1, log_interval=10, reset_num_timesteps=False)

# ============================== Dual Buffer 생성 ==============================

obs_shape = model.observation_space.shape
action_dim = model.action_space.shape[0]
human_buffer = CnnReplayBuffer(
    buffer_size=100000,
    observation_space=model.observation_space,
    action_space=model.action_space,
    device=model.device
)

agent_buffer = CnnReplayBuffer(
    buffer_size=1000000,
    observation_space=model.observation_space,
    action_space=model.action_space,
    device=model.device
)



# ============================== 키 입력 초기화 ==============================

# pygame.init()
# screen = pygame.display.set_mode((400, 300))
# pygame.display.set_caption("HIL Control Window")
current_steering, current_speed = 0.0, 0.0

# ============================== 개입 파라미터 ==============================

initial_alpha = 0.9
min_alpha = 0.0
decay_rate = 0.5
max_human_steps = 100000

# ============================== 사람 개입 액션 함수 ==============================

def get_human_action(original_action, step):
    global current_steering, current_speed
    keys = pygame.key.get_pressed()
    action = np.array(original_action, dtype=np.float32).reshape(-1)

    # 부드러운 변화값 설정
    steer_step = 0.03
    speed_step = 0.03
    brake_step = 0.1
    steering_recovery = 0.02
    max_steering = 0.5
    max_speed = 0.4

    # 🔁 조향 입력 (좌/우)
    if keys[pygame.K_LEFT]:
        current_steering -= steer_step
        action[2] = min(0.3, action[2] + brake_step)
    elif keys[pygame.K_RIGHT]:
        current_steering += steer_step
        action[2] = min(0.3, action[2] + brake_step)
    else:
        # 자동 복원 (방향키를 안 누르면 중앙으로 돌아옴)
        if current_steering > 0:
            current_steering = max(0, current_steering - steering_recovery)
        elif current_steering < 0:
            current_steering = min(0, current_steering + steering_recovery)

    # ⬆️ 가속
    if keys[pygame.K_UP]:
        current_speed += speed_step
        current_speed = min(current_speed, max_speed)
        action[2] = 0.0

    # ⬇️ 브레이크
    if keys[pygame.K_DOWN]:
        action[2] = 1.0
        current_speed *= 0.8

    # 브레이크 자동 감소
    if not keys[pygame.K_DOWN] and not keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
        action[2] = max(0.0, action[2] - 0.05)

    # ✅ 범위 제한
    current_steering = np.clip(current_steering, -max_steering, max_steering)
    current_speed = np.clip(current_speed, 0.0, 1.0)
    action[2] = np.clip(action[2], 0.0, 1.0)

    # 🤖 SAC + 인간 행동 혼합
    alpha = max(min_alpha, initial_alpha - decay_rate * (step / max_human_steps)) if step < max_human_steps else 0.0
    action[0] = alpha * current_steering + (1 - alpha) * action[0]
    action[1] = alpha * current_speed + (1 - alpha) * action[1]

    return action


# ============================== 학습 루프 ==============================

obs = env.reset()
obs = obs.transpose(0, 3, 1, 2)
done = False
total_timesteps = 1000000
step = 0
human_intervened_in_last_1000_steps = False

while step < total_timesteps:
    pygame.event.pump()
    human_override = False
    action = model.predict(obs, deterministic=True)[0]

    if step < max_human_steps and any(pygame.key.get_pressed()):
        action = get_human_action(action, step)
        human_override = True
        human_intervened_in_last_1000_steps = True

    action = np.array(action).reshape(1, -1)
    step_result = env.step(action)
    if len(step_result) == 4:
        next_obs, reward, done_flag, info = step_result
        terminated, truncated = done_flag, False
    else:
        next_obs, reward, terminated, truncated, info = step_result

    done = terminated or truncated
    next_obs = next_obs.transpose(0, 3, 1, 2)

    buffer_to_use = human_buffer if human_override else agent_buffer
    buffer_to_use.add(
        obs=np.array(obs),
        next_obs=np.array(next_obs),
        action=np.array(action),
        reward=np.array([reward]),
        done=np.array([terminated]),
        infos=[{}]
    )

    if step < max_human_steps and step % 1000 == 0 and human_intervened_in_last_1000_steps:
        print(f"📢 Step {step}: Human data collected. Skipping training for now.")
        human_intervened_in_last_1000_steps = False

    obs = next_obs
    step += 1
    env.render()
    print(f"Step: {step}, Human Override: {human_override}, Action: {action}, Reward: {reward}")

    if step % 10000 == 0 and step != 0:
        print(f"💾 Step {step}: 버퍼 자동 저장 중...")
        save_buffer(human_buffer, HUMAN_BUFFER_PATH)
        save_buffer(agent_buffer, AGENT_BUFFER_PATH)

    if step == max_human_steps:
        print("💾 사람 개입 모델 + 리플레이 버퍼 저장 중...")
        model.save(HUMAN_MODEL_PATH)
        human_buffer = load_buffer(HUMAN_BUFFER_PATH)
        agent_buffer = load_buffer(AGENT_BUFFER_PATH)


    if done:
        current_steering, current_speed = 0.0, 0.0
        obs = env.reset()
        obs = obs.transpose(0, 3, 1, 2)

# ============================== 10만 스텝 이후 학습 ==============================

print("🚀 사람 개입 데이터 기반으로 90만 스텝 학습을 시작합니다...")

# 모델 및 버퍼 로드
model = SAC.load(HUMAN_MODEL_PATH, env=env, tensorboard_log=LOG_DIR)
human_buffer.load(HUMAN_BUFFER_PATH)
agent_buffer.load(AGENT_BUFFER_PATH)

# 학습 루프 (900,000 스텝)
train_steps = 900000
batch_size = model.batch_size

for i in range(train_steps):
    # 사람 버퍼와 일반 버퍼에서 샘플링 비율 설정 (80:20)
    human_batch = human_buffer.sample(int(batch_size * 0.8), env=model._vec_normalize_env)
    agent_batch = agent_buffer.sample(batch_size - human_batch.observations.shape[0], env=model._vec_normalize_env)

    # 배치 합치기
    obs = np.concatenate([human_batch.observations, agent_batch.observations], axis=0)
    actions = np.concatenate([human_batch.actions, agent_batch.actions], axis=0)
    next_obs = np.concatenate([human_batch.next_observations, agent_batch.next_observations], axis=0)
    rewards = np.concatenate([human_batch.rewards, agent_batch.rewards], axis=0)
    dones = np.concatenate([human_batch.dones, agent_batch.dones], axis=0)

    # 텐서 변환
    obs = torch.tensor(obs).to(model.device)
    actions = torch.tensor(actions).to(model.device)
    next_obs = torch.tensor(next_obs).to(model.device)
    rewards = torch.tensor(rewards).to(model.device)
    dones = torch.tensor(dones).to(model.device)

    # 정책 업데이트
    model.policy.train()
    model.policy.set_training_mode(True)
    model._update_learning_rate(model.policy.optimizer)
    model.policy.train_on_batch(obs, actions, rewards, next_obs, dones)

    if (i + 1) % 10000 == 0:
        print(f"🔥 Custom Train Step: {i+1} / {train_steps}")

# ============================== 모델 저장 ==============================

model.save(MODEL_PATH)
print(f"💾 학습이 완료되었습니다. 모델이 '{MODEL_PATH}'에 저장되었습니다.")
pygame.quit()

