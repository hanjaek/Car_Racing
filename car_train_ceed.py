import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# ✅ 모델 및 로그 저장할 폴더 설정
MODEL_DIR = "ceed_model_v1"
LOG_DIR = "tensorboard_logs"  
MODEL_PATH = os.path.join(MODEL_DIR, "sac_car_racing_best")

# 폴더가 없으면 생성
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ✅ 트랙을 고정하는 SEED 값 설정
SEED = 1  # 트랙을 고정하기 위한 SEED 값

# ✅ 환경 생성 함수 (SEED 적용)
def make_env():
    def _init():
        env = gym.make("CarRacing-v3", domain_randomize=False, render_mode="rgb_array")
        env = Monitor(env, filename=os.path.join(LOG_DIR, "basic_seed1_v1_2.csv"))  
        env.reset(seed=SEED)  # ✅ 트랙 고정
        return env
    return _init

# ✅ CarRacing 환경 생성 (SEED 적용)
env = DummyVecEnv([make_env()])
env.seed(SEED)  # ✅ 벡터 환경에도 SEED 적용

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
        tensorboard_log=LOG_DIR  # ✅ SAC 자동 로그만 저장
    )

# ✅ 학습 수행 (항상 동일한 트랙에서 학습)
model.learn(total_timesteps=1000000, log_interval=10)

# ✅ 모델 저장
model.save(MODEL_PATH)
print(f"💾 학습이 완료되었습니다. 모델이 '{MODEL_PATH}'에 저장되었습니다.")
