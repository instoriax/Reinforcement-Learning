from stable_baselines3 import PPO
from client import connect

# MODEL = '/nas3/instoria/RL/final_project/final_project_env/log/PPO/circle_checkpoint/best_model_1780000.zip'
MODEL = '/nas3/instoria/RL/final_project/final_project_env/log/PPO/austria_checkpoint/best_model_780000.zip'

model = PPO.load(MODEL)
connect(model)