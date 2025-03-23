import os
import gymnasium as gym
import numpy as np
import pygame  
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.buffers import ReplayBuffer

# ✅ Custom ReplayBuffer with human intervention prioritization
class HumanPrioritizedReplayBuffer(ReplayBuffer):
    def _init_storage(self):
        super()._init_storage()
        self.human_flags = np.zeros(self.buffer_size, dtype=bool)

    def add(self, obs, next_obs, action, reward, done, infos):
        human_flag = infos[0].get("human", False)
        super().add(obs, next_obs, action, reward, done, infos)
        self.human_flags[self.pos - 1] = human_flag

    def sample(self, batch_size, env=None):
        human_indices = np.where(self.human_flags[:self.size])[0]
        other_indices = np.where(~self.human_flags[:self.size])[0]

        n_human = int(batch_size * 0.7)
        n_other = batch_size - n_human

        chosen = []
        if len(human_indices) > 0:
            chosen.extend(np.random.choice(human_indices, min(n_human, len(human_indices)), replace=False))
        if len(other_indices) > 0:
            chosen.extend(np.random.choice(other_indices, min(n_other, len(other_indices)), replace=False))

        chosen = np.array(chosen)
        return super()._get_samples(chosen, env=env)

# Pygame 초기화
pygame.init()
screen = pygame.display.set_mode((400, 300))  
pygame.display.set_caption("HIL Control Window")

MODEL_DIR = "sac_hil_model_v1"
LOG_DIR = "tensorboard_logs"
MODEL_PATH = os.path.join(MODEL_DIR, "sac_car_racing_best")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

SEED = 1

def make_env():
    def _init():
        env = gym.make("CarRacing-v3", domain_randomize=False, render_mode="human")
        env = Monitor(env, filename=os.path.join(LOG_DIR, "SAC_HIL_ceed2.csv"))  
        env.reset(seed=SEED)
        return env
    return _init

env = DummyVecEnv([make_env()])
env.seed(SEED)

try:
    model = SAC.load(MODEL_PATH, env=env, tensorboard_log=LOG_DIR)
    print(f"✅ 기존 모델을 불러와서 추가 학습합니다. ({MODEL_PATH})")
except:
    print("🚀 기존 모델이 없어서 새로 학습을 시작합니다.")
    model = SAC(
        "CnnPolicy",
        env,
        replay_buffer_class=HumanPrioritizedReplayBuffer,
        replay_buffer_kwargs=dict(buffer_size=1000000),
        learning_rate=3e-4,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=LOG_DIR
    )

current_steering = 0.0  
current_speed = 0.0     

initial_alpha = 0.9
min_alpha = 0.0  
decay_rate = 0.5  
max_human_steps = 100000

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
    if not keys[pygame.K_DOWN] and not keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
        action[2] = max(0.0, action[2] - 0.05)

    if current_speed < 0.02:  
        current_speed = 0.0

    current_steering = np.clip(current_steering, -1.0, 1.0)
    current_speed = np.clip(current_speed, 0.0, 1.0)  
    action[2] = np.clip(action[2], 0.0, 1.0)  

    if step >= max_human_steps:
        alpha = 0.0
    else:
        alpha = max(min_alpha, initial_alpha - decay_rate * (step / max_human_steps))

    action[0] = alpha * current_steering + (1 - alpha) * action[0]  
    action[1] = alpha * current_speed + (1 - alpha) * action[1]  
    action[2] = alpha * action[2] + (1 - alpha) * action[2]  

    return action

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
        next_obs, reward, done, info = step_result
        terminated, truncated = done, False  
    elif len(step_result) == 5:  
        next_obs, reward, terminated, truncated, info = step_result
    else:
        raise ValueError("Unexpected number of return values from env.step(action): {len(step_result)}")

    done = terminated or truncated
    next_obs = next_obs.transpose(0, 3, 1, 2)  

    model.replay_buffer.add(
        np.array(obs),
        np.array(next_obs),
        np.array(action),
        np.array([reward]),
        np.array([terminated]),
        [{"human": human_override}]
    )

    if step < max_human_steps and step % 1000 == 0:
        if human_intervened_in_last_1000_steps:
            print(f"📢 Step {step}: Training for 1000 steps due to human intervention...")
            model.learn(total_timesteps=1000, reset_num_timesteps=False)
            human_intervened_in_last_1000_steps = False  

    obs = next_obs  
    step += 1
    env.render()

    print(f"Step: {step}, Human Override: {human_override}, Action: {action}")

    if step == max_human_steps:
        print("💾 사람 개입 모델 + 리플레이 버퍼 저장 중...")
        model.save("sac_hil_model_v1/after_human_model.zip")
        model.save_replay_buffer("sac_hil_model_v1/human_buffer.pkl")

    if done:
        current_steering = 0.0
        current_speed = 0.0
        obs = env.reset()
        obs = obs.transpose(0, 3, 1, 2)
        continue

print("🚀 사람 개입 데이터 기반으로 90만 스텝 학습을 시작합니다...")
model = SAC.load("sac_hil_model_v1/after_human_model.zip", env=env, tensorboard_log=LOG_DIR)
model.load_replay_buffer("sac_hil_model_v1/human_buffer.pkl")
model.learn(total_timesteps=900000, reset_num_timesteps=False)

model.save(MODEL_PATH)
print(f"💾 학습이 완료되었습니다. 모델이 '{MODEL_PATH}'에 저장되었습니다.")

pygame.quit()
