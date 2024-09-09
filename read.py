import pickle
import numpy as np

class ObservationElement:
    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype

class ReplayElement:
    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype

class TaskUniformReplayBuffer:
    def __init__(self, save_dir, batch_size, timesteps, replay_capacity, action_shape, action_dtype, reward_shape, reward_dtype, update_horizon, observation_elements, extra_replay_elements):
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.replay_capacity = replay_capacity
        self.action_shape = action_shape
        self.action_dtype = action_dtype
        self.reward_shape = reward_shape
        self.reward_dtype = reward_dtype
        self.update_horizon = update_horizon
        self.observation_elements = observation_elements
        self.extra_replay_elements = extra_replay_elements

def read_replay_file(filename):
    with open(filename, 'rb') as file:
        replay_buffer = pickle.load(file)
        return replay_buffer

def main():
    filename = 'replay_buffer.pkl'  # 假设replay文件是序列化的二进制文件
    replay_buffer = read_replay_file(filename)

    # 打印replay_buffer中的一些信息
    print("Batch Size:", replay_buffer.batch_size)
    print("Timesteps:", replay_buffer.timesteps)
    print("Replay Capacity:", replay_buffer.replay_capacity)

    for elem in replay_buffer.observation_elements:
        print(f"Observation Element: {elem.name}, Shape: {elem.shape}, Dtype: {elem.dtype}")

if __name__ == "__main__":
    main()