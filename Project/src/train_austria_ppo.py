import gymnasium
from stable_baselines3 import PPO
from racecar_gym.env import RaceEnv
import torch
from utils import *
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, VecMonitor
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation

SCENARIO = 'austria_competition'
CHECKPOINT_DIR = './log/PPO/austria_checkpoint/'
device = torch.device('cuda:0')
RESET_WHEN_COLLISION = True

if __name__ == "__main__":
    def make_env():
        env = RaceEnv(
            scenario=SCENARIO,
            render_mode='rgb_array_birds_eye',
            reset_when_collision=RESET_WHEN_COLLISION, # Only work for 'austria_competition' and 'austria_competition_collisionStop'
        )
        env = ChannelLastObservation(env)
        env = GrayScaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, 84)
        return env

    CPU = 4
    env = SubprocVecEnv([lambda: make_env() for i in range(CPU)])
    env = VecFrameStack(env, 8, channels_order='last')
    env = VecMonitor(env)

    callback = TrainCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./log/PPO/austria_checkpoint/", device = device, use_sde=True)
    model.learn(total_timesteps=10000000, callback = callback)

