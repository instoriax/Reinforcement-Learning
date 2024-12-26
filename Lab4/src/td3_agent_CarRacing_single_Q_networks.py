import torch
import torch.nn as nn
import numpy as np
from base_agent import TD3BaseAgent
from models.CarRacing_model import ActorNetSimple, CriticNetSimple
from environment_wrapper.CarRacingEnv import CarRacingEnvironment
import random
from base_agent import OUNoiseGenerator, GaussianNoise

class CarRacingTD3Agent(TD3BaseAgent):
	def __init__(self, config):
		super(CarRacingTD3Agent, self).__init__(config)
		# initialize environment
		self.env = CarRacingEnvironment(N_frame=4, test=False)
		self.test_env = CarRacingEnvironment(N_frame=4, test=True)
		
		# behavior network
		self.actor_net = ActorNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.critic_net1 = CriticNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.actor_net.to(self.device)
		self.critic_net1.to(self.device)
		# target network
		self.target_actor_net = ActorNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.target_critic_net1 = CriticNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.target_actor_net.to(self.device)
		self.target_critic_net1.to(self.device)
		self.target_actor_net.load_state_dict(self.actor_net.state_dict())
		self.target_critic_net1.load_state_dict(self.critic_net1.state_dict())
		
		# set optimizer
		self.lra = config["lra"]
		self.lrc = config["lrc"]
		
		self.actor_opt = torch.optim.Adam(self.actor_net.parameters(), lr=self.lra)
		self.critic_opt1 = torch.optim.Adam(self.critic_net1.parameters(), lr=self.lrc)

		# choose Gaussian noise or OU noise

		# noise_mean = np.full(self.env.action_space.shape[0], 0.0, np.float32)
		# noise_std = np.full(self.env.action_space.shape[0], 1.0, np.float32)
		# self.noise = OUNoiseGenerator(noise_mean, noise_std)

		self.noise = GaussianNoise(self.env.action_space.shape[0], 0.0, 1.0)
		self.steering_noise = 0.2
		self.gas_noise = 0.1
		self.breaking_noise = 0.1
		self.steering_clip = 0.5
		self.gas_clip = 0.25
		self.breaking_clip = 0.25		
	
	def decide_agent_actions(self, state, sigma=0.0, brake_rate=0.015):
		### TODO ###
		# based on the behavior (actor) network and exploration noise
		# with torch.no_grad():
		# 	state = ???
		# 	action = actor_net(state) + sigma * noise

		# return action

		with torch.no_grad():
			state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
			action = self.actor_net(state, brake_rate=brake_rate).cpu().numpy().squeeze() + sigma * self.noise.generate()

		return action
		

	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
		### TODO ###
		### TD3 ###
		# 1. Clipped Double Q-Learning for Actor-Critic
		# 2. Delayed Policy Updates
		# 3. Target Policy Smoothing Regularization

		## Update Critic ##
		# critic loss
		# q_value1 = ???
		# q_value2 = ???
		# with torch.no_grad():
		# 	# select action a_next from target actor network and add noise for smoothing
		# 	a_next = ??? + noise

		# 	q_next1 = ???
		# 	q_next2 = ???
		# 	# select min q value from q_next1 and q_next2 (double Q learning)
		# 	q_target = ???

		q_value1 = self.critic_net1(state, action)
		with torch.no_grad():
			noise = torch.stack(
				((torch.empty_like(action[:, 0]).data.normal_(0, self.steering_noise).clamp(-self.steering_clip, self.steering_clip)),
				(torch.empty_like(action[:, 1]).data.normal_(0, self.gas_noise).clamp(-self.gas_clip, self.gas_clip)),
				(torch.empty_like(action[:, 2]).data.normal_(0, self.breaking_noise).clamp(-self.breaking_clip, self.breaking_clip))),
				dim=1,
			)
			a_next = self.target_actor_net(next_state) + noise
			a_next[:, 0] = a_next[:, 0].clamp(-1, 1)
			a_next[:, 1] = a_next[:, 1].clamp(0, 1)
			a_next[:, 2] = a_next[:, 2].clamp(0, 1)
   
			q_next1 = self.target_critic_net1(next_state, a_next)
			q_target = reward + self.gamma * q_next1 * (1 - done)	

		# critic loss function
		criterion = nn.MSELoss()
		critic_loss1 = criterion(q_value1, q_target)

		# optimize critic
		self.critic_net1.zero_grad()
		critic_loss1.backward()
		self.critic_opt1.step()

		## Delayed Actor(Policy) Updates ##
		# if self.total_time_step % self.update_freq == 0:
		# 	## update actor ##
		# 	# actor loss
		# 	# select action a from behavior actor network (a is different from sample transition's action)
		# 	# get Q from behavior critic network, mean Q value -> objective function
		# 	# maximize (objective function) = minimize -1 * (objective function)
		# 	action = ???
		# 	actor_loss = -1 * (???)
		# 	# optimize actor
		# 	self.actor_net.zero_grad()
		# 	actor_loss.backward()
		# 	self.actor_opt.step()

		if self.total_time_step % self.update_freq == 0:
			action = self.actor_net(state)
			actor_loss = -1 * self.critic_net1(state, action).mean()
			self.actor_net.zero_grad()
			actor_loss.backward()
			self.actor_opt.step()
		
	def load_and_evaluate(self, load_path):
		self.load(load_path)
		env = CarRacingEnvironment(test=True)
		print("==============================================")
		print("Evaluating...")
		all_rewards = []
		for episode in range(self.eval_episode):
			total_reward = 0
			state, infos = env.reset()
			for t in range(10000):
				action = self.decide_agent_actions(state)
				next_state, reward, terminates, truncates, _ = env.step(action)
				total_reward += reward
				state = next_state
				if terminates or truncates:
					print(
						'Episode: {}\tLength: {:3d}\tTotal reward: {:.2f}'
						.format(episode+1, t, total_reward))
					all_rewards.append(total_reward)
					break

		avg = sum(all_rewards) / self.eval_episode
		print(f"average score: {avg}")
		print("==============================================")
		return avg

	def update(self):
		# update the behavior networks
		self.update_behavior_network()
		# update the target networks
		if self.total_time_step % self.update_freq == 0:
			self.update_target_network(self.target_actor_net, self.actor_net, self.tau)
			self.update_target_network(self.target_critic_net1, self.critic_net1, self.tau)

	def save(self, save_path):
		torch.save(
				{
					'actor': self.actor_net.state_dict(),
					'critic1': self.critic_net1.state_dict(),
				}, save_path)

	def load(self, load_path):
		checkpoint = torch.load(load_path)
		self.actor_net.load_state_dict(checkpoint['actor'])
		self.critic_net1.load_state_dict(checkpoint['critic1'])