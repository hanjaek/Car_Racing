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

def find_last_event_file(log_dir):
    event_files = sorted(glob.glob(os.path.join(log_dir, "events.out.tfevents.*")))
    if not event_files:
        print(f"❌ No TensorBoard event file found in: {log_dir}")
        return None
    return event_files[-1]  # 마지막 로그 사용

# SAC+HIL 모델만
hil_model_list = [
    ("tensorboard_logs/sac_hil_model_1", "SAC+HIL Model 1", "orange"),
    ("tensorboard_logs/sac_hil_model_2", "SAC+HIL Model 2", "cyan"),
    ("tensorboard_logs/sac_hil_model_3", "SAC+HIL Model 3", "magenta"),
]

plt.figure(figsize=(12, 6))

for log_dir, label, color in hil_model_list:
    event_file = find_last_event_file(log_dir)
    if event_file is None:
        continue
    steps, rewards = load_tb_log(event_file)
    if steps is not None and rewards is not None:
        if len(rewards) >= 21:
            smoothed_rewards = savgol_filter(rewards, window_length=21, polyorder=3)
        else:
            smoothed_rewards = rewards
        plt.plot(steps, smoothed_rewards, label=label, color=color)

plt.title("SAC+HIL Reward Comparison")
plt.xlabel("Training Steps")
plt.ylabel("Episode Reward")
plt.xlim(0, 1_000_000)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("sac_hil_only_plot.png", dpi=300, bbox_inches='tight')
plt.show()
