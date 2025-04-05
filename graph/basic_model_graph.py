import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
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
        print(f"❌ No TensorBoard event file found in: {log_dir}")
        return None
    return event_files[0]

# SAC 모델 디렉토리 목록
sac_dirs = [
    "tensorboard_logs/basic_model_1",
    "tensorboard_logs/basic_model_2",
    "tensorboard_logs/basic_model_3"
]

colors = ['red', 'green', 'blue']
labels = ['SAC Model 1', 'SAC Model 2', 'SAC Model 3']

plt.figure(figsize=(10, 6))

for i, log_dir in enumerate(sac_dirs):
    event_file = safe_find_event_file(log_dir)
    if event_file is None:
        continue
    steps, rewards = load_tb_log(event_file)
    if steps is not None and rewards is not None:
        # 살살 smoothing 적용
        if len(rewards) >= 21:
            smoothed_rewards = savgol_filter(rewards, window_length=21, polyorder=3)
        else:
            smoothed_rewards = rewards  # 데이터가 짧으면 smoothing 안 함
        plt.plot(steps, smoothed_rewards, label=labels[i], color=colors[i])

plt.title("SAC Reward Comparison")
plt.xlabel("Training Steps")
plt.ylabel("Episode Reward")
plt.xlim(0, 1_000_000)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("SAC_reward_plot.png", dpi=300, bbox_inches='tight') 
plt.show()
