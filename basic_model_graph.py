import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_tb_log(file_path, tag='rollout/ep_rew_mean', max_step=1_000_000):
    ea = EventAccumulator(file_path)
    ea.Reload()
    if tag not in ea.Tags().get('scalars', []):
        print(f"⚠️ Tag '{tag}' not found in {file_path}")
        return None, None
    events = ea.Scalars(tag)
    steps = [e.step for e in events if e.step <= max_step]
    values = [e.value for e in events if e.step <= max_step]
    return np.array(steps), np.array(values)

def safe_find_event_file(log_dir):
    event_files = sorted(glob.glob(os.path.join(log_dir, "events.out.tfevents.*")))
    if not event_files:
        print(f"❌ No TensorBoard log file found in: {log_dir}")
        return None
    return event_files[0]

# SAC 모델 디렉토리 목록 (HIL 제외)
sac_dirs = [
    "tensorboard_logs/basic_model_1",
    "tensorboard_logs/basic_model_2",
    "tensorboard_logs/basic_model_3"
]

# 색상 및 라벨 지정
colors = ['gray', 'purple', 'orange']
labels = ['SAC 모델 1', 'SAC 모델 2', 'SAC 모델 3']

# 📈 시각화 시작
plt.figure(figsize=(10, 6))

for i, log_dir in enumerate(sac_dirs):
    event_file = safe_find_event_file(log_dir)
    if event_file is None:
        continue
    steps, rewards = load_tb_log(event_file)
    if steps is not None:
        plt.plot(steps, rewards, label=labels[i], color=colors[i])

plt.title("SAC 모델별 Reward 비교")
plt.xlabel("Training Step")
plt.ylabel("Episode Reward")
plt.xlim(0, 1_000_000)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
