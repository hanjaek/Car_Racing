import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
import matplotlib

# ✅ Matplotlib 백엔드 비활성화 (경고 메시지 방지)
matplotlib.use('Agg')

class CustomCarRacingEnv(gym.Wrapper):
    def __init__(self, env):
        super(CustomCarRacingEnv, self).__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 🎯 사용자 정의 보상 적용
        reward = self.custom_reward(obs, action)

        return obs, reward, terminated, truncated, info

    def custom_reward(self, obs, action):
        """
        도로 중심을 유지하고 정상적인 주행을 유도하는 보상 함수
        """
        speed_reward = obs[4]  # 속도 유지 보상
        track_position = obs[1]  # 트랙 중심에서의 거리
        angle = obs[3]  # 차량 방향과 도로 방향의 차이

        # 도로 중심 유지 보상
        track_reward = 1.0 - abs(track_position)

        # 도로 이탈 패널티
        off_road_penalty = -10 if abs(track_position) > 0.9 else 0

        # 차량 방향 패널티
        angle_penalty = -abs(angle)

        # 총 보상 계산
        total_reward = speed_reward + track_reward + off_road_penalty + angle_penalty
        return total_reward

# 🎯 환경 감싸기 (VecEnv 적용)
env = DummyVecEnv([lambda: CustomCarRacingEnv(gym.make("CarRacing-v3", render_mode="human"))])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)  # 관찰값과 보상 정규화

# ✅ 저장된 모델과 환경이 있는지 확인 후 불러오기
if os.path.exists("sac_CarRacing.zip") and os.path.exists("sac_CarRacing_env.pkl"):
    env = VecNormalize.load("sac_CarRacing_env.pkl", env)  # ✅ 저장된 환경 불러오기
    model = SAC.load("sac_CarRacing", env=env)  # ✅ 기존 모델 불러오기
    print("✅ 기존 학습된 모델과 환경 불러와서 추가 학습 진행")
else:
    print("🚀 저장된 모델이 없거나 환경이 손상됨. 새로 학습 시작")
    model = SAC("MlpPolicy", env, verbose=1)

# 추가 학습 진행 (이전 학습 내용을 유지하며 학습)
model.learn(total_timesteps=50000, log_interval=4)

# ✅ 모델과 환경 저장
model.save("sac_CarRacing")
env.save("sac_CarRacing_env.pkl")  # ✅ 환경 정보도 함께 저장

# ✅ 현재 환경의 렌더 모드 확인
print("✅ 현재 환경 렌더 모드:", env.get_attr('render_mode'))

# ✅ 주행 화면 보기 (VecEnv 내부 환경 직접 렌더링)
obs, info = env.reset()
while True:
    env.envs[0].render()  # ✅ VecEnv 내부 환경에서 직접 render() 호출
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
