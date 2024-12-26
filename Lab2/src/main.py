from dqn_agent_atari import AtariDQNAgent

if __name__ == '__main__':
    # my hyperparameters, you can change it as you like
    config = {
		"gpu": True,
		"training_steps": 30000000,
		"gamma": 0.99,
		"batch_size": 32,
		"eps_min": 0.1,
		"warmup_steps": 20000,
		"eps_decay": 1000000,
		"eval_epsilon": 0.01,
		"replay_buffer_capacity": 100000,
		"logdir": 'log/DQN/MsPacman/',
        # "logdir": 'log/DQN/Enduro/',
		"update_freq": 4,
		"update_target_freq": 10000,
		"learning_rate": 0.0000625,
        "eval_interval": 100,
        "eval_episode": 5,
        # "env_id": 'ALE/Enduro-v5',
		"env_id": 'ALE/MsPacman-v5',
	}
    agent = AtariDQNAgent(config)
    agent.load_and_evaluate("./model_MsPacman_5084.pth")
    # agent.load_and_evaluate("./model_Enduro_1431.pth")
    # agent.train()