import os
import glob
import numpy as np

outputs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
search_pattern = os.path.join(outputs_dir, "rollout_waypoint", "*", "*", "**", "episode.npz")
episodes = glob.glob(search_pattern, recursive=True)

def extract_datetime_key(filepath):
    parts = filepath.split(os.sep)
    idx = parts.index("rollout_waypoint")
    return (parts[idx+1], parts[idx+2])

latest_episode = max(episodes, key=extract_datetime_key)
print(f"Loading {latest_episode}")

data = np.load(latest_episode, allow_pickle=True)
for key in data.files:
    print(f"Key: {key}")
    print(data[key][:10])
