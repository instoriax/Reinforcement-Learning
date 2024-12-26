import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from base_agent import DQNBaseAgent
from models.atari_model import AtariNetDQN
import gym
import random

class AtariDQNAgent(DQNBaseAgent):
	def __init__(self, config):
		super(AtariDQNAgent, self).__init__(config)
		### TODO ###
		# initialize env
		# self.env = ???
		self.env = gym.make(config["env_id"], render_mode="rgb_array")
		self.env = gym.wrappers.RecordVideo(self.env, 'video')
		self.env = gym.wrappers.ResizeObservation(self.env, (84,84))
		self.env = gym.wrappers.GrayScaleObservation(self.env)
		self.env = gym.wrappers.FrameStack(self.env, 4)


		### TODO ###
		# initialize test_env
		# self.test_env = ???
		self.test_env = gym.make(config["env_id"], render_mode="rgb_array")
		self.test_env = gym.wrappers.RecordVideo(self.test_env, 'video')
		self.test_env = gym.wrappers.ResizeObservation(self.test_env, (84,84))
		self.test_env = gym.wrappers.GrayScaleObservation(self.test_env)
		self.test_env = gym.wrappers.FrameStack(self.test_env, 4)

		# initialize behavior network and target network
		self.behavior_net = AtariNetDQN(self.env.action_space.n)
		self.behavior_net.to(self.device)
		self.target_net = AtariNetDQN(self.env.action_space.n)
		self.target_net.to(self.device)
		self.target_net.load_state_dict(self.behavior_net.state_dict())
		# initialize optimizer
		self.lr = config["learning_rate"]
		self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4)
		
	def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
		### TODO ###
		# get action from behavior net, with epsilon-greedy selection
		
		# if random.random() < epsilon:
		# 	action = ???
		# else:
		# 	action = ???

		# return action

		if random.random() < epsilon:
			action = action_space.sample()
		else:
			obs = np.array(observation)
			state = torch.tensor(obs)
			state = state.unsqueeze(0).to(self.device)
			with torch.no_grad():
				q_values = self.behavior_net(state)
			action = torch.argmax(q_values).item()
		return action
	
	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
		
		### TODO ###
		# calculate the loss and update the behavior network
		# 1. get Q(s,a) from behavior net
		# 2. get max_a Q(s',a) from target net
		# 3. calculate Q_target = r + gamma * max_a Q(s',a)
		# 4. calculate loss between Q(s,a) and Q_target
		# 5. update behavior net

		
		# q_value = ???
		# with torch.no_grad():
			# q_next = ???

			# if episode terminates at next_state, then q_target = reward
			# q_target = ???
        
		
		# criterion = ???
		# loss = criterion(q_value, q_target)

		# self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)

		# self.optim.zero_grad()
		# loss.backward()
		# self.optim.step()

        # 1. get Q(s,a) from behavior net
		q_value = self.behavior_net(state)
		q_value = q_value.gather(1, action.to(torch.int64))

        # 2. get max_a Q(s',a) from target net
		with torch.no_grad():
			q_next = self.target_net(next_state)
			q_next, _ = torch.max(q_next, dim=1, keepdim=True)

        # 3. calculate Q_target = r + gamma * max_a Q(s',a)
		q_target = reward + self.gamma * q_next
		q_target[done.flatten()==1] = reward[done.flatten()==1]

        # 4. calculate loss between Q(s,a) and Q_target
		criterion = nn.MSELoss()
		loss = criterion(q_value, q_target)
		
        # 5. update behavior net
		self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)

		self.optim.zero_grad()
		loss.backward()
		self.optim.step()
	
	