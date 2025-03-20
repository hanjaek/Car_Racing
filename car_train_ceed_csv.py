import os
import csv
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

#  모델 및 로그 저장할 폴더 설정
MODEL_DIR = "ceed_model_v1"
LOG_DIR = "tensorboard_logs"  
MODEL_PATH = os.path.join(MODEL_DIR, "sac_car_racing_best")
CSV_FILE = "test.csv"  #  학습 데이터를 반영할 CSV 파일

# 폴더가 없으면 생성
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

#  트랙을 고정하는 SEED 값 설정
SEED = 1  # 트랙을 고정하기 위한 SEED 값

#  환경 생성 함수 (SEED 적용)
def make_env():
    def _init():
        env = gym.make("CarRacing-v3", domain_randomize=False, render_mode="rgb_array")
        env.reset(seed=SEED)  #  트랙 고정
        return env
    return _init

#  CarRacing 환경 생성 (SEED 적용)
env = DummyVecEnv([make_env()])
env.seed(SEED)  #  벡터 환경에도 SEED 적용

#  기존 모델 불러오기 or 새로운 모델 생성
try:
    model = SAC.load(MODEL_PATH, env=env, tensorboard_log=LOG_DIR)
    print(f" 기존 모델을 불러와서 추가 학습합니다. ({MODEL_PATH})")
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
        tensorboard_log=LOG_DIR  #  SAC 자동 로그만 저장
    )

#  `test.csv` 데이터를 SAC의 리플레이 버퍼에 추가하는 함수
def load_csv_to_replay_buffer(csv_file, model, env):
    """
    CSV 파일에서 데이터를 불러와 SAC의 리플레이 버퍼에 추가하는 함수
    """
    if not os.path.exists(csv_file):
        print(f"⚠ CSV 파일 '{csv_file}'이 존재하지 않습니다. 기존 학습만 진행합니다.")
        return
    
    with open(csv_file, mode="r") as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            step = int(row["Step"])
            action = np.array([[float(row["Steering"]), float(row["Acceleration"]), float(row["Brake"])]], dtype=np.float32)

            obs = env.reset()
            obs = obs.transpose(0, 3, 1, 2)  

            step_result = env.step(action)

            if len(step_result) == 4:  
                next_obs, reward, done, info = step_result
                terminated, truncated = done, False  
            elif len(step_result) == 5:  
                next_obs, reward, terminated, truncated, info = step_result
            else:
                raise ValueError(f"Unexpected number of return values from env.step(action): {len(step_result)}")

            next_obs = next_obs.transpose(0, 3, 1, 2)
            done = terminated or truncated

            #  리플레이 버퍼에 추가 (커브 데이터 반영)
            model.replay_buffer.add(
                np.array(obs),  
                np.array(next_obs),  
                np.array(action),  
                np.array([reward]),  
                np.array([done]),  
                [{}]  
            )

            print(f"📥 Step {step}: Action {action} (CSV 데이터 SAC 버퍼에 추가)")

    print(f" CSV 데이터를 SAC 리플레이 버퍼에 추가 완료!")

#  CSV 데이터 SAC 리플레이 버퍼에 추가
load_csv_to_replay_buffer(CSV_FILE, model, env)

#  CSV 데이터를 우선적으로 학습시키기 위해 5000 스텝 학습
print("🚀 CSV 데이터를 반영하여 모델을 5000 스텝 추가 학습합니다...")
model.learn(total_timesteps=5000)

#  기존 학습을 유지하며 300만 스텝 추가 학습
print("📢 SAC 모델을 기존 학습 방식으로 300만 스텝 추가 학습합니다...")
model.learn(total_timesteps=1000000)

#  모델 저장
model.save(MODEL_PATH)
print(f"💾 학습이 완료되었습니다. 모델이 '{MODEL_PATH}'에 저장되었습니다.")
