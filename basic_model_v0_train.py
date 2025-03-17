import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

# ✅ TensorBoard 콜백 (여러 가지 학습 지표 기록)
class TensorBoardCallback(BaseCallback):
    def __init__(self, log_dir):
        super().__init__()
        self.writer = SummaryWriter(log_dir)

    def _on_step(self) -> bool:
        # 매 1000 스텝마다 기록
        if self.n_calls % 1000 == 0:
            # SAC 모델에서 손실(loss) 값 가져오기
            actor_loss = self.model.actor.optimizer.param_groups[0]['lr']
            critic_loss = self.model.critic.optimizer.param_groups[0]['lr']
            entropy_loss = self.model.ent_coef_optimizer.param_groups[0]['lr']
            episode_reward = self.locals['rewards'][0]

            # TensorBoard에 기록
            self.writer.add_scalar("Loss/Actor Loss", actor_loss, self.num_timesteps)
            self.writer.add_scalar("Loss/Critic Loss", critic_loss, self.num_timesteps)
            self.writer.add_scalar("Loss/Entropy Loss", entropy_loss, self.num_timesteps)
            self.writer.add_scalar("Rewards/Episode Reward", episode_reward, self.num_timesteps)
            
        return True

# ✅ 모델 및 로그 저장할 폴더 설정
MODEL_DIR = "basic_model_v0"
LOG_DIR = "tensorboard_logs/basic_model_v0"
MODEL_PATH = os.path.join(MODEL_DIR, "sac_car_racing_best")

# 폴더가 없으면 생성
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ✅ CarRacing 환경 생성
env = gym.make("CarRacing-v3", domain_randomize=False, render_mode="rgb_array")
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

# ✅ 학습 수행 (100만 스텝)
model.learn(total_timesteps=1000000, callback=TensorBoardCallback(LOG_DIR))

# ✅ 모델 저장
model.save(MODEL_PATH)
print(f"💾 학습이 완료되었습니다. 모델이 '{MODEL_PATH}'에 저장되었습니다.")
