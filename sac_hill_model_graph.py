import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_and_merge_all_logs(log_dir, tag='rollout/ep_rew_mean', max_step=1_000_000):
    event_files = sorted(glob.glob(os.path.join(log_dir, "events.out.tfevents.*")))
    all_events = []

    for file_path in event_files:
        ea = EventAccumulator(file_path)
        ea.Reload()
        if tag not in ea.Tags().get('scalars', []):
            continue
        events = ea.Scalars(tag)
        all_events.extend([e for e in events if e.step <= max_step])

    if not all_events:
        print(f"⚠️ No events found in: {log_dir}")
        return None, None

    # step 순으로 정렬
    all_events.sort(key=lambda e: e.step)
    steps = [e.step for e in all_events]
    rewards = [e.value for e in all_events]
    return np.array(steps), np.array(rewards)

# SAC+HIL 모델만
hil_model_list = [
    ("tensorboard_logs/sac_hil_model_1", "SAC+HIL Model 1", "red"),
    ("tensorboard_logs/sac_hil_model_2", "SAC+HIL Model 2", "blue"),
    ("tensorboard_logs/sac_hil_model_3", "SAC+HIL Model 3", "green"),
]

plt.figure(figsize=(12, 6))

for log_dir, label, color in hil_model_list:
    steps, rewards = load_and_merge_all_logs(log_dir)
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
